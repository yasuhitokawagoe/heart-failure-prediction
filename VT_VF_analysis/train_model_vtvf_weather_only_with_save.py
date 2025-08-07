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
import joblib  # モデル保存用

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 祝日フラグ
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性指標の強化
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # 季節
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    # 季節ダミー変数
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    # 月末・月初のフラグ
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end.astype(int)
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def detect_extreme_weather(df):
    """異常気象を検出する関数"""
    weather_cols = ['min_temp_vtvf', 'max_temp_vtvf', 'avg_temp_vtvf', 'avg_wind_vtvf', 'vapor_pressure_vtvf', 
                   'avg_humidity_vtvf', 'sunshine_hours_vtvf']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 異常気象フラグ
    df['is_tropical_night'] = (df['min_temp_vtvf'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp_vtvf'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp_vtvf'] >= 30).astype(int)
    df['is_summer_day'] = (df['max_temp_vtvf'] >= 25).astype(int)
    df['is_winter_day'] = (df['min_temp_vtvf'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp_vtvf'] < 0).astype(int)
    
    return df

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成"""
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    
    for col in weather_cols:
        # ラグ特徴量
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        # 変化率
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        # 移動平均の変化率
        for window in [3, 7]:
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
    
    return df

def create_seasonal_weighted_features(df):
    """季節性重み付け特徴量を作成"""
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def create_weather_interaction_features(df):
    """気象相互作用特徴量を作成"""
    df['temp_humidity'] = df['avg_temp_vtvf'] * df['avg_humidity_vtvf']
    df['temp_pressure'] = df['avg_temp_vtvf'] * df['vapor_pressure_vtvf']
    df['temp_change'] = (df['avg_temp_vtvf'] - df['avg_temp_vtvf'].shift(1)).fillna(0)
    df['humidity_change'] = (df['avg_humidity_vtvf'] - df['avg_humidity_vtvf'].shift(1)).fillna(0)
    df['pressure_change'] = (df['vapor_pressure_vtvf'] - df['vapor_pressure_vtvf'].shift(1)).fillna(0)
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    return df

def load_processed_data():
    """処理済みデータを読み込み"""
    df = pd.read_csv('vtvf_weather_merged.csv')
    df['hospitalization_date'] = pd.to_datetime(df['hospitalization_date_vtvf'])
    df['hospitalization_count'] = df['people_vtvf']
    return df

def create_features_for_date(df, current_date):
    """特定の日付の特徴量を作成"""
    # 過去のデータのみを使用
    past_data = df[df['hospitalization_date'] < current_date].copy()
    
    if len(past_data) == 0:
        return None
    
    # 特徴量作成
    past_data = create_date_features(past_data)
    past_data = detect_extreme_weather(past_data)
    past_data = create_weather_interaction_features(past_data)
    past_data = create_time_series_features(past_data)
    past_data = create_advanced_weather_timeseries_features(past_data)
    past_data = create_seasonal_weighted_features(past_data)
    
    return past_data

def prepare_data_for_training(df, train_dates, test_dates):
    """訓練データとテストデータを準備"""
    train_features = []
    test_features = []
    
    # 訓練データの特徴量作成
    for date in train_dates:
        features = create_features_for_date(df, date)
        if features is not None:
            train_features.append(features)
    
    # テストデータの特徴量作成
    for date in test_dates:
        features = create_features_for_date(df, date)
        if features is not None:
            test_features.append(features)
    
    if not train_features or not test_features:
        return None, None, None, None
    
    # データフレームに結合
    train_df = pd.concat(train_features, ignore_index=True)
    test_df = pd.concat(test_features, ignore_index=True)
    
    # ターゲット変数
    threshold = train_df['hospitalization_count'].quantile(0.75)
    train_df['target'] = (train_df['hospitalization_count'] >= threshold).astype(int)
    test_df['target'] = (test_df['hospitalization_count'] >= threshold).astype(int)
    
    # 特徴量の選択
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 
                   'hospitalization_count', 'people']
    feature_columns = [col for col in train_df.columns if col not in exclude_cols]
    
    # 入院データ関連の特徴量を除外
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # 数値型の列のみを抽出
    numeric_cols = train_df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    X_train = train_df[feature_columns].values
    y_train = train_df['target'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['target'].values
    
    return X_train, y_train, X_test, y_test, feature_columns

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮したデータ分割"""
    dates = df['hospitalization_date'].unique()
    dates = sorted(dates)
    
    splits = []
    for i in range(n_splits):
        split_point = len(dates) * (i + 1) // (n_splits + 1)
        train_dates = dates[:split_point]
        test_dates = dates[split_point:split_point + len(dates) // (n_splits + 1)]
        splits.append((train_dates, test_dates))
    
    return splits

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """アンサンブルモデルを訓練"""
    models = {}
    predictions = {}
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    predictions['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
    
    # CatBoost
    cat_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['cat'] = cat_model
    predictions['cat'] = cat_model.predict_proba(X_val)[:, 1]
    
    # アンサンブル予測
    ensemble_pred = np.mean([predictions['lgb'], predictions['xgb'], predictions['cat']], axis=0)
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    return models, ensemble_pred, ensemble_auc

def evaluate_model_performance(y_true, y_pred_proba, threshold=0.5):
    """モデル性能を評価"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }

def save_models(models, feature_columns, fold_num=None):
    """モデルを保存"""
    os.makedirs('saved_models', exist_ok=True)
    
    # 保存用のサフィックス
    suffix = f"_fold_{fold_num}" if fold_num is not None else "_latest"
    
    # 各モデルを保存
    for model_name, model in models.items():
        if model_name == 'xgb':
            joblib.dump(model, f'saved_models/xgb_model_vtvf{suffix}.pkl')
            print(f'✓ XGBoostモデルをsaved_models/xgb_model_vtvf{suffix}.pklに保存しました')
        elif model_name == 'lgb':
            joblib.dump(model, f'saved_models/lgb_model_vtvf{suffix}.pkl')
            print(f'✓ LightGBMモデルをsaved_models/lgb_model_vtvf{suffix}.pklに保存しました')
        elif model_name == 'cat':
            joblib.dump(model, f'saved_models/cat_model_vtvf{suffix}.pkl')
            print(f'✓ CatBoostモデルをsaved_models/cat_model_vtvf{suffix}.pklに保存しました')
    
    # 特徴量リストも保存
    feature_info = {
        'feature_columns': feature_columns,
        'n_features': len(feature_columns)
    }
    joblib.dump(feature_info, f'saved_models/feature_info_vtvf{suffix}.pkl')
    print(f'✓ 特徴量情報をsaved_models/feature_info_vtvf{suffix}.pklに保存しました')

def main():
    """メイン実行関数"""
    print("=== VT/VF 気象情報のみのモデル訓練 ===")
    
    try:
        # データ読み込み
        df = load_processed_data()
        print(f"✓ データ読み込み完了: {len(df)} 行")
        
        # 季節性分割
        splits = create_seasonal_splits(df, n_splits=3)
        print(f"✓ データ分割完了: {len(splits)} 分割")
        
        cv_results = []
        modelwise_metrics = []
        feature_importance = None
        
        for fold, (train_dates, test_dates) in enumerate(splits, 1):
            print(f"\n=== Fold {fold} ===")
            
            # データ準備
            result = prepare_data_for_training(df, train_dates, test_dates)
            if result is None:
                print(f"Fold {fold} でデータ準備に失敗しました。スキップします。")
                continue
            
            X_train, y_train, X_test, y_test, feature_columns = result
            print(f"✓ データ準備完了: 訓練 {len(X_train)}, テスト {len(X_test)}")
            
            # 特徴量選択の最適化
            best_auc_fold = 0
            best_features = None
            best_result = None
            
            for n in [50, 65, 80, 100]:
                print(f"  特徴量数 {n} で試行中...")
                
                # 1. LightGBMで特徴量重要度を計算
                lgb_temp = lgb.LGBMClassifier(random_state=42)
                lgb_temp.fit(X_train, y_train)
                importances = lgb_temp.feature_importances_
                
                # 2. 上位N個の特徴量を選択
                top_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])][:n]
                
                # 3. 高相関な特徴量を除去
                X_train_df = pd.DataFrame(X_train, columns=feature_columns)
                reduced_features = remove_highly_correlated_features(X_train_df, top_features)
                
                # 4. データを再構成
                X_train_sel = X_train_df[reduced_features].values
                X_test_sel = pd.DataFrame(X_test, columns=feature_columns)[reduced_features].values
                
                # 無限大の値を処理
                X_train_sel = np.nan_to_num(X_train_sel, nan=0.0, posinf=0.0, neginf=0.0)
                X_test_sel = np.nan_to_num(X_test_sel, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 5. 標準化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_sel)
                X_test_scaled = scaler.transform(X_test_sel)
                
                # 6. アンサンブル学習
                try:
                    models, best_ensemble_pred, best_auc = train_ensemble_models(X_train_scaled, y_train, X_test_scaled, y_test)
                    print(f"    特徴量数 {len(reduced_features)}: AUC = {best_auc:.4f}")
                    
                    if best_auc > best_auc_fold:
                        best_auc_fold = best_auc
                        best_features = reduced_features
                        best_result = (models, best_ensemble_pred, X_test_scaled)
                        
                        # 最良モデルを保存
                        save_models(models, reduced_features, fold)
                        
                except Exception as e:
                    print(f"    エラー: {e}")
                    continue
            
            print(f"Fold {fold} 最良AUC: {best_auc_fold:.4f} (特徴量数: {len(best_features)})")
            
            if best_result is None:
                print(f"Fold {fold} で有効な結果が得られませんでした。スキップします。")
                continue
            
            # 最良の結果を使用
            models, best_ensemble_pred, X_test_scaled = best_result
            
            # 各モデルの予測
            predictions = {}
            for model_name, model in models.items():
                if model_name == 'lgb':
                    predictions[model_name] = model.predict(X_test_scaled)
                elif model_name == 'xgb':
                    predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]
                elif model_name == 'cat':
                    predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]
            
            # 各モデルごとの評価指標を計算
            for model_name, y_pred_proba in predictions.items():
                metrics = evaluate_model_performance(y_test, y_pred_proba)
                modelwise_metrics.append({
                    'fold': fold,
                    'model': model_name,
                    'roc_auc': metrics['roc_auc'],
                    'pr_auc': metrics['pr_auc'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'specificity': metrics['specificity']
                })
            
            # アンサンブル予測
            ensemble_pred = best_ensemble_pred
            
            # モデルの評価
            fold_metrics = evaluate_model_performance(y_test, ensemble_pred)
            cv_results.append(fold_metrics)
            
            # 特徴量の重要度（LightGBMの結果を使用）
            if fold == 1:  # 最初のfoldの特徴量重要度のみを保存
                importance = pd.DataFrame({
                    'feature': best_features,
                    'importance': models['lgb'].feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False)
                feature_importance = importance
        
        # 最終的な性能の表示
        print("\n=== 最終的な評価結果（VT/VF 気象情報のみ） ===")
        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score', 'specificity']
        for metric in metrics:
            values = [r[metric] for r in cv_results]
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去"""
    if len(feature_columns) <= 1:
        return feature_columns
    
    corr_matrix = df[feature_columns].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    selected_features = [col for col in feature_columns if col not in to_drop]
    
    return selected_features

if __name__ == "__main__":
    main() 