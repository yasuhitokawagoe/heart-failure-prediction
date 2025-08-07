#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全気象予測モデル 自動最適化版 再開用
中断したポイントから再開するためのスクリプト
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import optuna
from optuna.samplers import TPESampler
import jpholiday
import tensorflow as tf
from scipy.optimize import minimize
import itertools
import warnings
warnings.filterwarnings('ignore')

def create_hf_specific_date_features(df):
    """心不全特化の日付特徴量を作成"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week
    
    # 心不全に影響する曜日パターン
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)  # 月曜日は心不全増加
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)  # 金曜日も注意
    
    # 祝日・連休の影響
    df['is_holiday'] = df['date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性（心不全は冬に悪化しやすい）
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 冬期フラグ（心不全悪化期）
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
    
    # 月末・月初（医療機関の混雑期）
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    return df

def create_hf_specific_weather_features(df):
    """心不全特化の気象特徴量を作成"""
    # 基本気象データの前処理
    weather_cols = ['min_temp_weather', 'max_temp_weather', 'avg_temp_weather', 
                   'avg_wind_weather', 'pressure_local', 'avg_humidity_weather', 
                   'sunshine_hours_weather', 'precipitation']
    
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 心不全に影響する温度変化
    df['temp_range'] = df['max_temp_weather'] - df['min_temp_weather']  # 日較差
    df['temp_change_from_yesterday'] = df['avg_temp_weather'].diff()  # 前日比
    df['temp_change_3day'] = df['avg_temp_weather'].diff(3)  # 3日前比
    
    # 心不全悪化のリスク要因
    df['is_cold_stress'] = (df['min_temp_weather'] < 5).astype(int)  # 寒冷ストレス
    df['is_heat_stress'] = (df['max_temp_weather'] > 30).astype(int)  # 暑熱ストレス
    df['is_temperature_shock'] = (abs(df['temp_change_from_yesterday']) > 10).astype(int)  # 急激な温度変化
    
    # 湿度関連（心不全患者は湿度に敏感）
    df['is_high_humidity'] = (df['avg_humidity_weather'] > 80).astype(int)
    df['is_low_humidity'] = (df['avg_humidity_weather'] < 30).astype(int)
    df['humidity_change'] = df['avg_humidity_weather'].diff()
    
    # 気圧関連（心不全患者は気圧変化に敏感）
    df['pressure_change'] = df['pressure_local'].diff()
    df['pressure_change_3day'] = df['pressure_local'].diff(3)
    df['is_pressure_drop'] = (df['pressure_change'] < -5).astype(int)  # 気圧低下
    df['is_pressure_rise'] = (df['pressure_change'] > 5).astype(int)   # 気圧上昇
    
    # 風関連
    df['is_strong_wind'] = (df['avg_wind_weather'] > 10).astype(int)
    df['wind_change'] = df['avg_wind_weather'].diff()
    
    # 降水量関連
    df['is_rainy'] = (df['precipitation'] > 0).astype(int)
    df['is_heavy_rain'] = (df['precipitation'] > 50).astype(int)
    df['rain_days_consecutive'] = df['is_rainy'].rolling(window=7).sum()
    
    return df

def create_hf_advanced_features(df):
    """心不全特化の高度な特徴量を作成"""
    
    # 移動平均・標準偏差（心不全の慢性経過を反映）
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        df[f'{col}_ma_3'] = df[col].rolling(window=3).mean()
        df[f'{col}_ma_7'] = df[col].rolling(window=7).mean()
        df[f'{col}_ma_14'] = df[col].rolling(window=14).mean()
        df[f'{col}_std_7'] = df[col].rolling(window=7).std()
    
    # 季節性を考慮した重み付け特徴量
    df['winter_temp_weighted'] = df['avg_temp_weather'] * df['is_winter']
    df['summer_temp_weighted'] = df['avg_temp_weather'] * df['is_summer']
    
    # 心不全悪化リスクの複合指標
    df['hf_risk_score'] = (
        df['is_cold_stress'] * 2 + 
        df['is_heat_stress'] * 1.5 + 
        df['is_temperature_shock'] * 3 + 
        df['is_pressure_drop'] * 2 + 
        df['is_high_humidity'] * 1
    )
    
    # 気象ストレスの累積効果
    df['weather_stress_cumulative'] = df['hf_risk_score'].rolling(window=7).sum()
    
    # 急激な変化の検出
    df['temp_acceleration'] = df['temp_change_from_yesterday'].diff()
    df['pressure_acceleration'] = df['pressure_change'].diff()
    
    return df

def create_hf_interaction_features(df):
    """心不全特化の相互作用特徴量を作成"""
    
    # 温度×湿度の相互作用
    df['temp_humidity_interaction'] = df['avg_temp_weather'] * df['avg_humidity_weather']
    df['temp_humidity_ratio'] = df['avg_temp_weather'] / (df['avg_humidity_weather'] + 1)
    
    # 温度×気圧の相互作用
    df['temp_pressure_interaction'] = df['avg_temp_weather'] * df['pressure_local']
    
    # 季節×気象の相互作用
    df['winter_temp'] = df['avg_temp_weather'] * df['is_winter']
    df['summer_humidity'] = df['avg_humidity_weather'] * df['is_summer']
    
    # 曜日×気象の相互作用
    df['monday_temp'] = df['avg_temp_weather'] * df['is_monday']
    df['weekend_pressure'] = df['pressure_local'] * df['is_weekend']
    
    return df

def convert_weather_to_binary(df):
    """天気概況を二値変数に変換する関数"""
    # 天気概況のカラムを確認
    weather_cols = [col for col in df.columns if '天気概況' in col or '天気分類' in col]
    
    for col in weather_cols:
        if col in df.columns:
            # 各天気条件を二値変数に変換
            unique_weather = df[col].dropna().unique()
            
            for weather in unique_weather:
                if pd.notna(weather) and weather != '':
                    # 天気名を安全なカラム名に変換
                    safe_name = str(weather).replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                    col_name = f"{col}_{safe_name}"
                    df[col_name] = (df[col] == weather).astype(int)
    
    # 元の天気概況カラムを削除（二値変数に変換済み）
    for col in weather_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df

def optimize_hf_hyperparameters(X_train, y_train, X_val, y_val):
    """心不全特化のハイパーパラメータ最適化"""
    
    def objective(trial):
        # LightGBMの最適化
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42
        }
        
        # XGBoostの最適化
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 10.0),
            'random_state': 42
        }
        
        # CatBoostの最適化
        cb_params = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'iterations': trial.suggest_int('cb_iterations', 100, 1000),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('cb_depth', 3, 12),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1.0, 10.0),
            'random_state': 42
        }
        
        # モデル学習
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        xgb_model = xgb.XGBClassifier(**xgb_params)
        cb_model = cb.CatBoostClassifier(**cb_params, verbose=False)
        
        # 予測
        lgb_pred = lgb_model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
        xgb_pred = xgb_model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
        cb_pred = cb_model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
        
        # アンサンブル予測
        ensemble_pred = (lgb_pred + xgb_pred + cb_pred) / 3
        
        # エラーハンドリング: 1つのクラスのみの場合
        try:
            auc_score = roc_auc_score(y_val, ensemble_pred)
            return auc_score
        except ValueError as e:
            if "Only one class present" in str(e):
                # 1つのクラスのみの場合は低いスコアを返す
                return 0.5
            else:
                raise e
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    
    # 進捗表示付きで最適化
    print(f"    最適化開始: 50回の試行")
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"    最適化完了: 最高AUC = {study.best_value:.4f}")
    
    return study.best_params

def create_hf_ensemble_models(X_train, y_train, X_val, y_val, best_params):
    """心不全特化のアンサンブルモデルを作成"""
    
    # 最適化されたパラメータでモデル作成
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        n_estimators=best_params.get('n_estimators', 500),
        learning_rate=best_params.get('learning_rate', 0.1),
        max_depth=best_params.get('max_depth', 6),
        num_leaves=best_params.get('num_leaves', 50),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        reg_alpha=best_params.get('reg_alpha', 0.1),
        reg_lambda=best_params.get('reg_lambda', 0.1),
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=best_params.get('xgb_n_estimators', 500),
        learning_rate=best_params.get('xgb_learning_rate', 0.1),
        max_depth=best_params.get('xgb_max_depth', 6),
        subsample=best_params.get('xgb_subsample', 0.8),
        colsample_bytree=best_params.get('xgb_colsample_bytree', 0.8),
        reg_alpha=best_params.get('xgb_reg_alpha', 0.1),
        reg_lambda=best_params.get('xgb_reg_lambda', 0.1),
        random_state=42
    )
    
    cb_model = cb.CatBoostClassifier(
        objective='Logloss',
        eval_metric='AUC',
        iterations=best_params.get('cb_iterations', 500),
        learning_rate=best_params.get('cb_learning_rate', 0.1),
        depth=best_params.get('cb_depth', 6),
        l2_leaf_reg=best_params.get('cb_l2_leaf_reg', 3.0),
        random_state=42,
        verbose=False
    )
    
    # モデル学習
    lgb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)
    
    # 予測
    lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
    xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
    cb_pred = cb_model.predict_proba(X_val)[:, 1]
    
    # 動的重み付け
    weights = optimize_ensemble_weights([lgb_pred, xgb_pred, cb_pred], y_val)
    ensemble_pred = (weights[0] * lgb_pred + weights[1] * xgb_pred + weights[2] * cb_pred)
    
    return [lgb_model, xgb_model, cb_model], ensemble_pred, weights

def optimize_ensemble_weights(predictions, y_true):
    """アンサンブル重みの最適化"""
    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 正規化
        ensemble_pred = np.sum([w * pred for w, pred in zip(weights, predictions)], axis=0)
        try:
            return -roc_auc_score(y_true, ensemble_pred)
        except ValueError as e:
            if "Only one class present" in str(e):
                # 1つのクラスのみの場合は低いスコアを返す
                return 0.0
            else:
                raise e
    
    # 初期重み
    initial_weights = [1/3, 1/3, 1/3]
    
    # 最適化
    result = minimize(objective, initial_weights, method='L-BFGS-B', 
                     bounds=[(0, 1)] * len(predictions))
    
    optimal_weights = result.x / np.sum(result.x)  # 正規化
    return optimal_weights

def create_seasonal_splits(df, n_splits=20):
    """季節性を考慮した時系列分割を作成（HF特化版）"""
    splits = []
    unique_dates = pd.Series(df['date'].unique()).sort_values()
    total_days = len(unique_dates)
    
    # 各分割のサイズを計算
    test_size = 30  # 1ヶ月のテストデータ
    train_size = 365  # 1年の訓練データ
    
    # 年間を通じてテスト期間を分散させる
    for i in range(n_splits):
        # テスト期間の開始位置を計算（年間を通じて分散）
        monthly_offset = (i % 12) * 30  # 0-11ヶ月分のオフセット
        year_offset = (i // 12) * 365   # 年単位のオフセット
        
        base_offset = monthly_offset + year_offset
        
        # テスト期間の設定
        test_end_idx = total_days - base_offset
        test_start_idx = test_end_idx - test_size
        
        # 学習期間の設定（テスト期間の直前）
        train_end_idx = test_start_idx - 1
        train_start_idx = max(0, train_end_idx - train_size)
        
        # 期間の日付を取得
        test_dates = unique_dates.iloc[test_start_idx:test_end_idx]
        train_dates = unique_dates.iloc[train_start_idx:train_end_idx]
        
        # データの準備
        X_train, y_train, X_test, y_test, feature_columns = prepare_data_for_training(df, train_dates, test_dates)
        
        splits.append((X_train, y_train, X_test, y_test, feature_columns))
        
        print(f"Split {i+1}:")
        print(f"訓練期間: {train_dates.min()} から {train_dates.max()}")
        print(f"テスト期間: {test_dates.min()} から {test_dates.max()}\n")
    
    return splits

def prepare_data_for_training(df, train_dates, test_dates):
    """トレーニングデータとテストデータを準備（気象情報のみ）"""
    # データを日付で分割
    train_data = df[df['date'].isin(train_dates)].copy()
    test_data = df[df['date'].isin(test_dates)].copy()
    
    # 特徴量とターゲットを分離
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 'hospitalization_count', 'people_hf', 'people_weather']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院データに関連する特徴量を完全に除外
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # カテゴリカル変数をダミー変数に変換
    categorical_cols = ['season']
    for col in categorical_cols:
        if col in df.columns:
            # ダミー変数を作成
            dummies = pd.get_dummies(df[col], prefix=col)
            # 元のカラムを削除
            feature_columns = [f for f in feature_columns if f != col]
            # ダミー変数を追加
            feature_columns.extend(dummies.columns)
            # データにダミー変数を追加
            train_data = pd.concat([train_data, dummies.loc[train_data.index]], axis=1)
            test_data = pd.concat([test_data, dummies.loc[test_data.index]], axis=1)
    
    # 数値型の列のみを抽出
    numeric_cols = train_data[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    # 訓練データの標準化
    scaler = StandardScaler()
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    
    # テストデータの標準化（訓練データのスケーラーを使用）
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])
    
    X_train = train_data[feature_columns]
    y_train = train_data['target']
    X_test = test_data[feature_columns]
    y_test = test_data['target']
    
    return X_train, y_train, X_test, y_test, feature_columns

def load_processed_data():
    """データの読み込み"""
    df = pd.read_csv('hf_weather_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def save_prediction_details(X_test, y_test, predictions, fold_idx, results_dir):
    """予測詳細を保存する関数"""
    prediction_data = {
        'actual': y_test,
        'predicted_prob': predictions,
        'fold': fold_idx
    }
    
    df_predictions = pd.DataFrame(prediction_data)
    df_predictions.to_csv(f'{results_dir}/predictions_fold_{fold_idx}.csv', index=False)
    
    return df_predictions

def load_existing_results(results_dir):
    """既存の結果を読み込む"""
    existing_results = []
    existing_predictions = []
    
    # 既存の予測ファイルを確認
    for i in range(1, 21):  # 1-20フォルド
        pred_file = f'{results_dir}/predictions_fold_{i}.csv'
        if os.path.exists(pred_file):
            pred_df = pd.read_csv(pred_file)
            existing_predictions.append(pred_df)
            print(f"既存の結果を読み込み: Fold {i}")
        else:
            break
    
    return existing_results, existing_predictions

def main():
    """メイン実行関数（再開用）"""
    import time
    
    print("心不全気象予測モデル 自動最適化版 再開用")
    print("=" * 50)
    
    start_time = time.time()
    
    # 結果保存ディレクトリ
    results_dir = '心不全気象予測モデル_自動最適化版_結果'
    os.makedirs(results_dir, exist_ok=True)
    
    # 既存の結果を読み込み
    existing_results, existing_predictions = load_existing_results(results_dir)
    completed_folds = len(existing_predictions)
    
    print(f"完了済みフォルド: {completed_folds}/20")
    
    if completed_folds >= 20:
        print("全てのフォルドが完了しています。結果を統合します。")
        # 結果統合処理
        all_predictions_df = pd.concat(existing_predictions, ignore_index=True)
        all_predictions_df.to_csv(f'{results_dir}/all_predictions.csv', index=False)
        
        # 全体性能計算
        overall_auc = roc_auc_score(all_predictions_df['actual'], all_predictions_df['predicted_prob'])
        overall_pr_auc = average_precision_score(all_predictions_df['actual'], all_predictions_df['predicted_prob'])
        
        print(f"\n{'='*50}")
        print(f"=== 最終結果 ===")
        print(f"全体AUC: {overall_auc:.4f}")
        print(f"全体PR-AUC: {overall_pr_auc:.4f}")
        return
    
    # データの読み込み
    df = load_processed_data()
    print(f"データ読み込み完了: {len(df)} 件")
    
    # ターゲット変数の作成
    threshold = df['people_hf'].quantile(0.75)
    df['target'] = (df['people_hf'] >= threshold).astype(int)
    print(f"ターゲット変数作成: 閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
    
    # HF特化の特徴量作成
    print("HF特化特徴量を作成中...")
    df = create_hf_specific_date_features(df)
    df = create_hf_specific_weather_features(df)
    df = create_hf_advanced_features(df)
    df = create_hf_interaction_features(df)
    df = convert_weather_to_binary(df)
    
    # 特徴量選択
    exclude_cols = ['date', 'people_hf', 'target', 'season']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    print(f"使用特徴量数: {len(feature_columns)}")
    
    # 欠損値処理
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 時系列分割
    splits = create_seasonal_splits(df, n_splits=20)
    
    # 評価結果保存
    cv_results = existing_results.copy()
    all_predictions = existing_predictions.copy()
    
    print(f"\n再開: Fold {completed_folds + 1} から開始")
    print(f"予想時間: 約{(20 - completed_folds) * 5}分")
    
    for fold, (X_train, y_train, X_test, y_test, feature_columns) in enumerate(splits[completed_folds:], completed_folds + 1):
        fold_start_time = time.time()
        print(f"\n{'='*20} Fold {fold}/20 {'='*20}")
        
        # ハイパーパラメータ最適化
        best_params = optimize_hf_hyperparameters(X_train, y_train, X_test, y_test)
        
        # アンサンブルモデル作成
        models, ensemble_pred, weights = create_hf_ensemble_models(X_train, y_train, X_test, y_test, best_params)
        
        # 性能評価
        try:
            auc_score = roc_auc_score(y_test, ensemble_pred)
            pr_auc = average_precision_score(y_test, ensemble_pred)
        except ValueError as e:
            if "Only one class present" in str(e):
                # 1つのクラスのみの場合は低いスコアを返す
                auc_score = 0.5
                pr_auc = 0.5
                print(f"    警告: テストデータに1つのクラスのみ含まれています。AUC=0.5を設定します。")
            else:
                raise e
        
        # 結果保存
        fold_result = {
            'fold': fold,
            'auc': auc_score,
            'pr_auc': pr_auc,
            'weights': weights.tolist(),
            'best_params': best_params
        }
        cv_results.append(fold_result)
        
        # 予測詳細保存
        pred_details = save_prediction_details(X_test, y_test, ensemble_pred, fold, results_dir)
        all_predictions.append(pred_details)
        
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} 完了: AUC = {auc_score:.4f}, PR-AUC = {pr_auc:.4f}")
        print(f"Fold {fold} 所要時間: {fold_time:.1f}秒")
        
        # 残り時間の推定
        remaining_folds = 20 - fold
        avg_fold_time = fold_time
        estimated_remaining = remaining_folds * avg_fold_time
        print(f"推定残り時間: {estimated_remaining/60:.1f}分")
    
    # 全予測データを統合
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(f'{results_dir}/all_predictions.csv', index=False)
    
    # 結果保存
    with open(f'{results_dir}/optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)
    
    # 全体性能計算
    overall_auc = roc_auc_score(all_predictions_df['actual'], all_predictions_df['predicted_prob'])
    overall_pr_auc = average_precision_score(all_predictions_df['actual'], all_predictions_df['predicted_prob'])
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"=== 最終結果 ===")
    print(f"全体AUC: {overall_auc:.4f}")
    print(f"全体PR-AUC: {overall_pr_auc:.4f}")
    print(f"平均AUC: {np.mean([r['auc'] for r in cv_results]):.4f} ± {np.std([r['auc'] for r in cv_results]):.4f}")
    print(f"総実行時間: {total_time/60:.1f}分")
    
    # 結果サマリー保存
    summary = {
        'overall_auc': overall_auc,
        'overall_pr_auc': overall_pr_auc,
        'mean_auc': np.mean([r['auc'] for r in cv_results]),
        'std_auc': np.std([r['auc'] for r in cv_results]),
        'feature_count': len(feature_columns),
        'fold_count': len(cv_results),
        'total_time_minutes': total_time/60
    }
    
    with open(f'{results_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果を保存しました: {results_dir}/")

if __name__ == "__main__":
    main() 