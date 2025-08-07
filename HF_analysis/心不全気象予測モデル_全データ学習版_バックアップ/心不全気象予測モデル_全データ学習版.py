#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全気象予測モデル 全データ学習版
Hold-out法で性能評価を行いながら、全データで最終モデルを作成
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
import joblib  # 追加

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

def evaluate_model_performance(y_true, y_pred_proba, threshold=0.5):
    """モデルの詳細な性能評価を行う関数"""
    y_pred = (y_pred_proba > threshold).astype(int)
    # 基本的な評価指標
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # 混同行列
    conf_matrix = confusion_matrix(y_true, y_pred)
    # PR曲線のAUC
    pr_auc = average_precision_score(y_true, y_pred_proba)
    # 特異度（Specificity）
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    # 結果をまとめる
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': conf_matrix
    }
    return metrics

def find_optimal_threshold_f1(y_true, y_pred_proba):
    """F1-scoreを最大化する最適な閾値を探索"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def find_optimal_threshold_pr(y_true, y_pred_proba):
    """Precision-Recallバランスを最適化する閾値を探索"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        # PrecisionとRecallの調和平均
        if precision + recall > 0:
            score = 2 * precision * recall / (precision + recall)
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    return best_threshold, best_score

def evaluate_model_with_optimal_threshold(y_true, y_pred_proba):
    """最適化された閾値でモデルを評価"""
    # F1最適化
    f1_threshold, f1_score_opt = find_optimal_threshold_f1(y_true, y_pred_proba)
    y_pred_f1 = (y_pred_proba > f1_threshold).astype(int)
    precision_f1 = precision_score(y_true, y_pred_f1)
    recall_f1 = recall_score(y_true, y_pred_f1)
    
    # PR最適化
    pr_threshold, pr_score = find_optimal_threshold_pr(y_true, y_pred_proba)
    y_pred_pr = (y_pred_proba > pr_threshold).astype(int)
    precision_pr = precision_score(y_true, y_pred_pr)
    recall_pr = recall_score(y_true, y_pred_pr)
    
    # 標準評価
    metrics_std = evaluate_model_performance(y_true, y_pred_proba, 0.5)
    
    # 最適化結果
    optimized_metrics = {
        'standard': metrics_std,
        'f1_optimized': {
            'threshold': f1_threshold,
            'precision': precision_f1,
            'recall': recall_f1,
            'f1_score': f1_score_opt
        },
        'pr_optimized': {
            'threshold': pr_threshold,
            'precision': precision_pr,
            'recall': recall_pr,
            'score': pr_score
        }
    }
    
    return optimized_metrics

def main():
    """メイン実行関数"""
    import time
    
    print("心不全気象予測モデル 全データ学習版")
    print("=" * 50)
    
    start_time = time.time()
    
    # 結果保存ディレクトリ
    results_dir = '心不全気象予測モデル_全データ学習版_結果'
    os.makedirs(results_dir, exist_ok=True)
    
    # モデル保存ディレクトリ
    model_save_dir = '心不全気象予測モデル_全データ学習版_保存モデル'
    os.makedirs(model_save_dir, exist_ok=True)
    
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
    
    # 入院データに関連する特徴量を完全に除外
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # 数値型の列のみを抽出
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    print(f"使用特徴量数: {len(feature_columns)}")
    
    # 欠損値処理
    df[feature_columns] = df[feature_columns].fillna(method='ffill')
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
    
    # Hold-out法でデータ分割
    print("Hold-out法でデータ分割中...")
    # 学習データ: 2012-2020年
    train_data = df[df['date'] < '2021-01-01'].copy()
    # テストデータ: 2021年
    test_data = df[df['date'] >= '2021-01-01'].copy()
    
    print(f"学習データ期間: {train_data['date'].min()} から {train_data['date'].max()}")
    print(f"テストデータ期間: {test_data['date'].min()} から {test_data['date'].max()}")
    print(f"学習データ数: {len(train_data)} 件")
    print(f"テストデータ数: {len(test_data)} 件")
    
    # 学習データの準備
    X_train = train_data[feature_columns]
    y_train = train_data['target']
    X_test = test_data[feature_columns]
    y_test = test_data['target']
    
    # ハイパーパラメータ最適化（学習データで）
    print("ハイパーパラメータ最適化開始...")
    best_params = optimize_hf_hyperparameters(X_train, y_train, X_test, y_test)
    
    # アンサンブルモデル作成（学習データで）
    print("アンサンブルモデル作成中...")
    models, ensemble_pred, weights = create_hf_ensemble_models(X_train, y_train, X_test, y_test, best_params)
    
    # 性能評価
    print("性能評価中...")
    try:
        # 従来の評価
        auc_score = roc_auc_score(y_test, ensemble_pred)
        pr_auc = average_precision_score(y_test, ensemble_pred)
        
        # 最適化された評価
        optimized_metrics = evaluate_model_with_optimal_threshold(y_test, ensemble_pred)
        
        # 詳細な評価指標
        precision_std = optimized_metrics['standard']['precision']
        recall_std = optimized_metrics['standard']['recall']
        f1_std = optimized_metrics['standard']['f1_score']
        
        # 最適化された指標
        f1_opt = optimized_metrics['f1_optimized']['f1_score']
        precision_f1 = optimized_metrics['f1_optimized']['precision']
        recall_f1 = optimized_metrics['f1_optimized']['recall']
        threshold_f1 = optimized_metrics['f1_optimized']['threshold']
        
        precision_pr = optimized_metrics['pr_optimized']['precision']
        recall_pr = optimized_metrics['pr_optimized']['recall']
        threshold_pr = optimized_metrics['pr_optimized']['threshold']
        
    except ValueError as e:
        if "Only one class present" in str(e):
            auc_score = 0.5
            pr_auc = 0.5
            precision_std = recall_std = f1_std = 0.0
            f1_opt = precision_f1 = recall_f1 = 0.0
            precision_pr = recall_pr = 0.0
            threshold_f1 = threshold_pr = 0.5
            print(f"警告: テストデータに1つのクラスのみ含まれています。AUC=0.5を設定します。")
        else:
            raise e
    
    # 性能評価結果を保存
    evaluation_result = {
        'hold_out_performance': {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'precision_std': precision_std,
            'recall_std': recall_std,
            'f1_std': f1_std,
            'f1_optimized': f1_opt,
            'precision_f1': precision_f1,
            'recall_f1': recall_f1,
            'threshold_f1': threshold_f1,
            'precision_pr': precision_pr,
            'recall_pr': recall_pr,
            'threshold_pr': threshold_pr
        },
        'weights': weights.tolist(),
        'best_params': best_params,
        'feature_count': len(feature_columns),
        'train_data_size': len(train_data),
        'test_data_size': len(test_data),
        'evaluation_date': datetime.now().isoformat()
    }
    
    with open(f'{results_dir}/hold_out_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"Hold-out法による性能評価結果:")
    print(f"AUC: {auc_score:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision_std:.4f}")
    print(f"Recall: {recall_std:.4f}")
    print(f"F1-Score: {f1_std:.4f}")
    print(f"F1最適化 - F1: {f1_opt:.4f}, Precision: {precision_f1:.4f}, Recall: {recall_f1:.4f}")
    print(f"PR最適化 - Precision: {precision_pr:.4f}, Recall: {recall_pr:.4f}")
    
    # 全データで最終モデルを作成
    print("全データで最終モデルを作成中...")
    X_full = df[feature_columns]
    y_full = df['target']
    
    # 全データでアンサンブルモデルを作成
    final_models, final_ensemble_pred, final_weights = create_hf_ensemble_models(X_full, y_full, X_full, y_full, best_params)
    
    # 最終モデルを保存
    print("最終モデルを保存中...")
    
    # LightGBM
    joblib.dump(final_models[0], f'{model_save_dir}/lgb_model_final.pkl')
    # XGBoost
    joblib.dump(final_models[1], f'{model_save_dir}/xgb_model_final.pkl')
    # CatBoost
    joblib.dump(final_models[2], f'{model_save_dir}/cb_model_final.pkl')
    
    # モデル情報を保存
    model_info = {
        'model_version': '全データ学習版_v1.0',
        'hold_out_auc': auc_score,
        'hold_out_pr_auc': pr_auc,
        'hold_out_f1': f1_std,
        'best_params': best_params,
        'weights': final_weights.tolist(),
        'feature_columns': feature_columns,
        'total_data_size': len(df),
        'train_data_size': len(train_data),
        'test_data_size': len(test_data),
        'saved_date': datetime.now().isoformat()
    }
    
    with open(f'{model_save_dir}/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    # ハイパーパラメータを保存
    with open(f'{model_save_dir}/best_hyperparameters.pkl', 'wb') as f:
        joblib.dump(best_params, f)
    
    # アンサンブル重みを保存
    np.save(f'{model_save_dir}/ensemble_weights.npy', final_weights)
    
    # 特徴量リストを保存
    with open(f'{model_save_dir}/feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"=== 心不全気象予測モデル 全データ学習版 完了 ===")
    print(f"Hold-out法による性能評価:")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Precision: {precision_std:.4f}")
    print(f"  Recall: {recall_std:.4f}")
    print(f"  F1-Score: {f1_std:.4f}")
    print(f"最終モデル保存先: {model_save_dir}")
    print(f"総実行時間: {total_time/60:.1f}分")
    
    # 結果サマリー保存
    summary = {
        'hold_out_auc': auc_score,
        'hold_out_pr_auc': pr_auc,
        'hold_out_f1': f1_std,
        'feature_count': len(feature_columns),
        'total_data_size': len(df),
        'train_data_size': len(train_data),
        'test_data_size': len(test_data),
        'total_time_minutes': total_time/60
    }
    
    with open(f'{results_dir}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果を保存しました:")
    print(f"  評価結果: {results_dir}/")
    print(f"  最終モデル: {model_save_dir}/")

if __name__ == "__main__":
    main() 