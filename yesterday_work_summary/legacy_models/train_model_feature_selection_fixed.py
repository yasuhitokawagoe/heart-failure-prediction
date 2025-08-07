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

def create_basic_features(df):
    """基本特徴量のみを作成"""
    # 日付関連の基本特徴量
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 祝日フラグ
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性指標
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    return df

def create_weather_features(df):
    """気象特徴量を追加"""
    # 基本気象特徴量
    weather_cols = ['min_temp', 'max_temp', 'avg_temp', 'avg_wind', 'vapor_pressure', 
                   'avg_humidity', 'sunshine_hours']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 気象変化率
    df['temp_change'] = (df['avg_temp'] - df['avg_temp'].shift(1)).fillna(0)
    df['humidity_change'] = (df['avg_humidity'] - df['avg_humidity'].shift(1)).fillna(0)
    df['pressure_change'] = (df['vapor_pressure'] - df['vapor_pressure'].shift(1)).fillna(0)
    
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    return df

def create_extreme_weather_features(df):
    """異常気象特徴量を追加"""
    # 熱帯夜
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    
    # 猛暑日
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    
    # 真夏日
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    
    # 冬日
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    
    return df

def create_time_series_features(df, train_data=None):
    """時系列特徴量を追加（リーク防止版）"""
    # 移動平均
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    windows = [7, 14, 30]
    
    for col in weather_cols:
        for window in windows:
            if train_data is not None:
                # 訓練データの統計量を使用してテストデータの特徴量を作成
                train_stats = train_data[col].rolling(window=window, min_periods=window).mean()
                last_train_mean = train_stats.iloc[-1] if not train_stats.empty else df[col].mean()
                
                train_std = train_data[col].rolling(window=window, min_periods=window).std()
                last_train_std = train_std.iloc[-1] if not train_std.empty else df[col].std()
                
                df[f'{col}_ma_{window}d'] = last_train_mean
                df[f'{col}_std_{window}d'] = last_train_std
            else:
                # 訓練データ作成時は通常通り
                df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
                df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    # 変化率
    for col in weather_cols:
        if train_data is not None:
            # 訓練データの最後の値を使用
            last_train_value = train_data[col].iloc[-1] if not train_data[col].empty else df[col].mean()
            df[f'{col}_change_rate'] = (df[col] - last_train_value) / last_train_value if last_train_value != 0 else 0
            
            # 週間変化率（7日前の訓練データの値を使用）
            if len(train_data) >= 7:
                week_ago_value = train_data[col].iloc[-7]
                df[f'{col}_weekly_change_rate'] = (df[col] - week_ago_value) / week_ago_value if week_ago_value != 0 else 0
            else:
                df[f'{col}_weekly_change_rate'] = 0
        else:
            # 訓練データ作成時は通常通り
            df[f'{col}_change_rate'] = (df[col] - df[col].shift(1)) / df[col].shift(1)
            df[f'{col}_weekly_change_rate'] = (df[col] - df[col].shift(7)) / df[col].shift(7)
    
    return df

def load_processed_data():
    """処理済みデータを読み込み"""
    try:
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people']
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮した時系列分割を作成"""
    splits = []
    unique_dates = pd.Series(df['hospitalization_date'].unique()).sort_values()
    total_days = len(unique_dates)
    
    # 各分割のサイズを計算
    test_size = 30  # 1ヶ月のテストデータ
    train_size = 365  # 1年の訓練データ
    
    for i in range(n_splits):
        # テスト期間の設定（最新のデータから順に）
        test_end_idx = total_days - (i * test_size)
        test_start_idx = test_end_idx - test_size
        
        # 学習期間の設定（テスト期間の直前）
        train_end_idx = test_start_idx - 1  # テスト期間の直前まで
        train_start_idx = max(0, train_end_idx - train_size)
        
        # 期間の日付を取得
        test_dates = unique_dates.iloc[test_start_idx:test_end_idx]
        train_dates = unique_dates.iloc[train_start_idx:train_end_idx]
        
        splits.append((train_dates, test_dates))
        
        print(f"Split {i+1}:")
        print(f"訓練期間: {train_dates.min()} から {train_dates.max()}")
        print(f"テスト期間: {test_dates.min()} から {test_dates.max()}\n")
    
    return splits

def prepare_data_for_training(df, train_dates, test_dates, feature_creation_func):
    """トレーニングデータとテストデータを準備（リーク防止版）"""
    # データを日付で分割
    train_data = df[df['hospitalization_date'].isin(train_dates)].copy()
    test_data = df[df['hospitalization_date'].isin(test_dates)].copy()
    
    # 訓練データに特徴量を作成
    train_data = feature_creation_func(train_data)
    
    # テストデータに特徴量を作成（訓練データの情報のみを使用）
    if 'time_series' in str(feature_creation_func):
        # 時系列特徴量の場合は特別な処理
        test_data = create_basic_features(test_data)
        test_data = create_weather_features(test_data)
        test_data = create_extreme_weather_features(test_data)
        test_data = create_time_series_features(test_data, train_data)
    else:
        # 基本特徴量の場合は通常通り
        test_data = feature_creation_func(test_data)
    
    # 特徴量とターゲットを分離
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date']
    feature_columns = [col for col in train_data.columns if col not in exclude_cols]
    
    # 数値型の列のみを抽出
    numeric_cols = train_data[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    # 欠損値の処理
    train_data[feature_columns] = train_data[feature_columns].fillna(method='ffill')
    test_data[feature_columns] = test_data[feature_columns].fillna(method='ffill')
    
    train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
    test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
    
    # 無限大の値を適切な値に置換
    train_data[feature_columns] = train_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    test_data[feature_columns] = test_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    
    train_data[feature_columns] = train_data[feature_columns].fillna(0)
    test_data[feature_columns] = test_data[feature_columns].fillna(0)
    
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

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """LightGBMモデルを学習"""
    # 基本的なパラメータ
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # データセット作成
    train_data = lgb.Dataset(X_train, y_train)
    
    # モデル学習
    model = lgb.train(params, train_data, num_boost_round=1000)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    auc_score = roc_auc_score(y_test, y_pred)
    
    return model, y_pred, auc_score

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
        'specificity': specificity
    }
    
    return metrics

def main():
    """メイン実行関数"""
    try:
        # データの読み込み
        df = load_processed_data()
        
        # 日付でソート
        df = df.sort_values('hospitalization_date')
        
        print(f"全期間データ使用、データ数: {len(df)}")
        
        # ターゲット変数の作成（75%タイルで閾値を設定）
        threshold = df['hospitalization_count'].quantile(0.75)  # 75%タイル
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        print(f"ターゲット変数作成完了: 75%タイル閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
        
        # 特徴量セットの定義
        feature_sets = {
            'basic': lambda df: create_basic_features(df),
            'weather': lambda df: create_weather_features(create_basic_features(df)),
            'extreme_weather': lambda df: create_extreme_weather_features(create_weather_features(create_basic_features(df))),
            'time_series': lambda df: create_time_series_features(create_extreme_weather_features(create_weather_features(create_basic_features(df))))
        }
        
        # 時系列分割の作成
        splits = create_seasonal_splits(df, n_splits=3)
        
        # 各特徴量セットで実験
        results = {}
        
        for feature_set_name, feature_creation_func in feature_sets.items():
            print(f"\n=== {feature_set_name} 特徴量セットの実験 ===")
            
            # 各分割でモデルを学習・評価
            fold_results = []
            
            for fold, (train_dates, test_dates) in enumerate(splits, 1):
                print(f"Fold {fold}の処理を開始します...")
                
                # データの準備（リーク防止版）
                X_train, y_train, X_test, y_test, feature_columns = prepare_data_for_training(
                    df, train_dates, test_dates, feature_creation_func
                )
                
                # LightGBMモデルの学習
                model, y_pred, auc_score = train_lightgbm_model(X_train, y_train, X_test, y_test)
                
                # 評価
                metrics = evaluate_model_performance(y_test, y_pred)
                fold_results.append(metrics)
                
                print(f"Fold {fold} - AUC: {auc_score:.4f}")
            
            # 平均性能を計算
            avg_auc = np.mean([r['roc_auc'] for r in fold_results])
            avg_pr_auc = np.mean([r['pr_auc'] for r in fold_results])
            
            results[feature_set_name] = {
                'avg_auc': avg_auc,
                'avg_pr_auc': avg_pr_auc,
                'fold_results': fold_results
            }
            
            print(f"{feature_set_name} - 平均AUC: {avg_auc:.4f}, 平均PR-AUC: {avg_pr_auc:.4f}")
        
        # 結果の保存
        os.makedirs('feature_selection_results', exist_ok=True)
        
        # JSON保存用にデータを変換
        results_for_json = {}
        for key, value in results.items():
            results_for_json[key] = {
                'avg_auc': float(value['avg_auc']),
                'avg_pr_auc': float(value['avg_pr_auc']),
                'fold_results': [
                    {
                        'roc_auc': float(r['roc_auc']),
                        'pr_auc': float(r['pr_auc']),
                        'precision': float(r['precision']),
                        'recall': float(r['recall']),
                        'f1_score': float(r['f1_score']),
                        'specificity': float(r['specificity'])
                    }
                    for r in value['fold_results']
                ]
            }
        
        with open('feature_selection_results/results.json', 'w') as f:
            json.dump(results_for_json, f, indent=4)
        
        # 結果の表示
        print("\n=== 特徴量選択実験結果 ===")
        for feature_set_name, result in results.items():
            print(f"{feature_set_name}: AUC {result['avg_auc']:.4f}, PR-AUC {result['avg_pr_auc']:.4f}")
        
        # 最良の結果を特定
        best_feature_set = max(results.keys(), key=lambda x: results[x]['avg_auc'])
        print(f"\n最良の特徴量セット: {best_feature_set} (AUC: {results[best_feature_set]['avg_auc']:.4f})")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 