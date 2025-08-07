import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
import joblib

def encode_simplified_weather_conditions(df):
    """天気概況を6分類に簡略化して数値化"""
    # 天気分類(統合)を6分類に再マッピング
    weather_mapping = {
        '晴れ': '晴れ系',
        '快晴': '晴れ系',
        '曇り': '曇り系', 
        '薄曇': '曇り系',
        '小雨': '小雨',
        '大雨': '大雨',
        '雷雨': '雷雨',
        '雪': '雪'
    }
    
    # 天気分類(統合)を6分類に変換
    df['weather_simplified'] = df['天気分類(統合)'].map(weather_mapping)
    
    # 欠損値を'曇り系'で補完（最も多いカテゴリ）
    df['weather_simplified'] = df['weather_simplified'].fillna('曇り系')
    
    # LabelEncoderで数値化
    le = LabelEncoder()
    df['weather_simplified_encoded'] = le.fit_transform(df['weather_simplified'])
    
    print("=== 簡略化された天気分類 ===")
    print(f"分類数: {len(le.classes_)}")
    print(f"カテゴリ: {list(le.classes_)}")
    
    # 各分類の出現回数を確認
    print("\n=== 各分類の出現回数 ===")
    for category in le.classes_:
        count = (df['weather_simplified'] == category).sum()
        print(f"{category}: {count}回")
    
    return df, le

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
    # 月の周期性を考慮
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 日の周期性を考慮
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    
    # 曜日の周期性を考慮
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # 季節（3ヶ月ごと）
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
    
    # 四半期
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def detect_extreme_weather(df):
    """異常気象を検出する関数"""
    # NaN値を前方補完（未来のデータは使用しない）
    weather_cols = ['min_temp', 'max_temp', 'avg_temp', 'avg_wind', 'vapor_pressure', 
                   'avg_humidity', 'sunshine_hours']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    
    # 残りのNaN値を中央値で補完
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 熱帯夜（夜間の最低気温が25℃以上）
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    
    # 猛暑日（最高気温が35℃以上）
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    
    # 真夏日（最高気温が30℃以上）
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    
    # 夏日（最高気温が25℃以上）
    df['is_summer_day'] = (df['max_temp'] >= 25).astype(int)
    
    # 冬日（最低気温が0℃未満）
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    
    # 真冬日（最高気温が0℃未満）
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    
    # 寒波（その月の平均気温から大きく低い）
    df['monthly_temp_mean'] = df.groupby(['year', 'month'])['avg_temp'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )
    df['is_cold_wave'] = (df['avg_temp'] <= (df['monthly_temp_mean'] - 2.0)).astype(int)
    
    # 強風（平均風速が高い）
    df['wind_quantile'] = df['avg_wind'].expanding().quantile(0.95).shift(1).fillna(df['avg_wind'].quantile(0.95))
    df['is_strong_wind'] = (df['avg_wind'] > df['wind_quantile']).astype(int)
    
    # 台風の可能性（気圧が低く、風速が強い状態）
    df['pressure_quantile'] = df['vapor_pressure'].expanding().quantile(0.1).shift(1).fillna(df['vapor_pressure'].quantile(0.1))
    df['wind_quantile_90'] = df['avg_wind'].expanding().quantile(0.9).shift(1).fillna(df['avg_wind'].quantile(0.9))
    df['is_typhoon_condition'] = ((df['vapor_pressure'] < df['pressure_quantile']) & 
                                 (df['avg_wind'] > df['wind_quantile_90'])).astype(int)
    
    # 急激な気圧変化（前日との気圧差が大きい）
    df['pressure_change'] = df['vapor_pressure'] - df['vapor_pressure'].shift(1)
    df['pressure_change'] = df['pressure_change'].fillna(0)
    df['pressure_change_std'] = df['pressure_change'].expanding().std().shift(1).fillna(df['pressure_change'].std())
    df['is_rapid_pressure_change'] = (abs(df['pressure_change']) > df['pressure_change_std'] * 2).astype(int)
    
    # 異常湿度（極端に湿度が高いまたは低い）
    df['humidity_quantile_95'] = df['avg_humidity'].expanding().quantile(0.95).shift(1).fillna(df['avg_humidity'].quantile(0.95))
    df['humidity_quantile_05'] = df['avg_humidity'].expanding().quantile(0.05).shift(1).fillna(df['avg_humidity'].quantile(0.05))
    df['is_extreme_humidity_high'] = (df['avg_humidity'] > df['humidity_quantile_95']).astype(int)
    df['is_extreme_humidity_low'] = (df['avg_humidity'] < df['humidity_quantile_05']).astype(int)
    
    # 日照時間の極端な状態
    df['sunshine_quantile_95'] = df['sunshine_hours'].expanding().quantile(0.95).shift(1).fillna(df['sunshine_hours'].quantile(0.95))
    df['sunshine_quantile_05'] = df['sunshine_hours'].expanding().quantile(0.05).shift(1).fillna(df['sunshine_hours'].quantile(0.05))
    df['is_extremely_sunny'] = (df['sunshine_hours'] > df['sunshine_quantile_95']).astype(int)
    df['is_extremely_cloudy'] = (df['sunshine_hours'] < df['sunshine_quantile_05']).astype(int)
    
    # 気温の急激な変化（3日間で5℃以上の変化）
    df['temp_change_3d'] = df['avg_temp'] - df['avg_temp'].shift(3)
    df['is_rapid_temp_change'] = (abs(df['temp_change_3d']) > 5).astype(int)
    
    # 気温の変動性（7日間の標準偏差）
    df['temp_volatility'] = df['avg_temp'].rolling(window=7, min_periods=1).std().shift(1).fillna(0)
    df['temp_volatility_quantile'] = df['temp_volatility'].expanding().quantile(0.9).shift(1).fillna(df['temp_volatility'].quantile(0.9))
    df['is_high_temp_volatility'] = (df['temp_volatility'] > df['temp_volatility_quantile']).astype(int)
    
    # 複合的な気象ストレス指標
    df['weather_stress'] = (
        df['is_extremely_hot'].astype(float) * 0.3 +
        df['is_cold_wave'].astype(float) * 0.3 +
        df['is_rapid_pressure_change'].astype(float) * 0.2 +
        df['is_strong_wind'].astype(float) * 0.2
    )
    
    # 季節性を考慮した気象影響
    df['seasonal_weather_impact'] = (
        (df['month'].isin([6, 7, 8]) & df['is_extremely_hot']).astype(int) * 0.5 +
        (df['month'].isin([12, 1, 2]) & df['is_cold_wave']).astype(int) * 0.5
    )
    
    # 連続的な異常気象の検出（3日連続）
    extreme_conditions = ['is_tropical_night', 'is_extremely_hot', 'is_cold_wave', 
                         'is_strong_wind', 'is_typhoon_condition']
    
    for condition in extreme_conditions:
        condition_shifted1 = df[condition].shift(1).fillna(0)
        condition_shifted2 = df[condition].shift(2).fillna(0)
        df[f'{condition}_consecutive'] = ((df[condition] == 1) & 
                                        (condition_shifted1 == 1) & 
                                        (condition_shifted2 == 1)).astype(int)
    
    # 気温と湿度の複合的な影響
    df['temp_quantile_80'] = df['avg_temp'].expanding().quantile(0.8).shift(1).fillna(df['avg_temp'].quantile(0.8))
    df['humidity_quantile_80'] = df['avg_humidity'].expanding().quantile(0.8).shift(1).fillna(df['avg_humidity'].quantile(0.8))
    df['is_high_temp_high_humidity'] = ((df['avg_temp'] > df['temp_quantile_80']) & 
                                       (df['avg_humidity'] > df['humidity_quantile_80'])).astype(int)
    
    return df

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure', 'pressure_local', 'sunshine_hours']
    
    for col in weather_cols:
        # ラグ特徴量
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        # 変化率
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        # 移動平均とその変化率
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].rolling(window=window, min_periods=window).std()
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
        
        # より長期の移動平均
        for window in [30, 60, 90]:
            df[f'{col}_ma_{window}d'] = df[col].rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].rolling(window=window, min_periods=window).std()
        
        # トレンド特徴量
        df[f'{col}_trend_strength'] = df[f'{col}'].rolling(30).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 異常スコア
        df[f'{col}_anomaly_score'] = (df[col] - df[f'{col}_ma_30d']) / df[f'{col}_std_30d']
        
        # 季節性偏差
        for window in [7, 14, 30]:
            df[f'{col}_seasonal_deviation_{window}d'] = df[col] - df[f'{col}_ma_{window}d']
        
        # 加速度（2次微分）
        df[f'{col}_acceleration'] = df[f'{col}_change_rate'].diff()
        df[f'{col}_acceleration_2nd'] = df[f'{col}_acceleration'].diff()
        
        # 予測可能性（自己相関）
        df[f'{col}_predictability'] = df[col].rolling(7).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 2 else 0
        )
        
        # ボラティリティ
        df[f'{col}_volatility'] = df[f'{col}_std_7d'] / df[f'{col}_ma_7d']
        
        # 範囲特徴量
        for window in [3, 7, 14, 30]:
            df[f'{col}_range_{window}d'] = df[col].rolling(window).max() - df[col].rolling(window).min()
    
    # 変化率の組み合わせ
    df['temp_change_humidity_change'] = df['avg_temp_change_rate'] * df['avg_humidity_change_rate']
    df['temp_change_pressure_change'] = df['avg_temp_change_rate'] * df['vapor_pressure_change_rate']
    df['humidity_change_pressure_change'] = df['avg_humidity_change_rate'] * df['vapor_pressure_change_rate']
    
    # 非線形相互作用
    df['nonlinear_temp_pressure'] = df['avg_temp'] ** 2 * df['vapor_pressure']
    df['temp_humidity_pressure'] = df['avg_temp'] * df['avg_humidity'] * df['vapor_pressure']
    
    return df

def create_seasonal_weighted_features(df):
    """季節性重み付け特徴量を作成"""
    # 季節性重み
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure', 'pressure_local', 'sunshine_hours']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def create_weather_interaction_features(df):
    """気象相互作用特徴量を作成"""
    # 基本的な相互作用
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['vapor_pressure']
    df['humidity_pressure'] = df['avg_humidity'] * df['vapor_pressure']
    
    # 変化量
    df['temp_change'] = df['avg_temp'].diff()
    df['humidity_change'] = df['avg_humidity'].diff()
    df['pressure_change'] = df['vapor_pressure'].diff()
    
    # 変化量の相互作用
    df['temp_change_humidity_change'] = df['temp_change'] * df['humidity_change']
    df['temp_change_pressure_change'] = df['temp_change'] * df['pressure_change']
    df['humidity_change_pressure_change'] = df['humidity_change'] * df['pressure_change']
    
    # 不快指数の計算
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    # 異常気象の検出（元のモデルと同じ）
    df = detect_extreme_weather(df)
    
    # 異常気象と他の気象要素との相互作用
    weather_conditions = ['is_tropical_night', 'is_extremely_hot', 'is_cold_wave', 
                         'is_strong_wind', 'is_typhoon_condition']
    
    for condition in weather_conditions:
        # 気温との相互作用
        df[f'{condition}_temp_effect'] = df[condition].astype(float) * df['avg_temp']
        
        # 湿度との相互作用
        df[f'{condition}_humidity_effect'] = df[condition].astype(float) * df['avg_humidity']
        
        # 気圧との相互作用
        df[f'{condition}_pressure_effect'] = df[condition].astype(float) * df['vapor_pressure']
    
    # 季節性を考慮した異常気象の影響
    df['summer_extreme_heat'] = ((df['season'] == 'summer') & 
                                ((df['is_extremely_hot'] == 1) | (df['is_tropical_night'] == 1))).astype(int)
    df['winter_extreme_cold'] = ((df['season'] == 'winter') & 
                                ((df['is_cold_wave'] == 1) | (df['is_freezing_day'] == 1))).astype(int)
    
    # 連続的な異常気象の検出
    for condition in weather_conditions:
        # シフトされた値のNaNを0で補完
        condition_shifted1 = df[condition].shift(1).fillna(0)
        condition_shifted2 = df[condition].shift(2).fillna(0)
        
        # 3日連続で異常気象が続いているかどうか
        df[f'{condition}_consecutive'] = ((df[condition] == 1) & 
                                        (condition_shifted1 == 1) & 
                                        (condition_shifted2 == 1)).astype(int)
    
    # 複合的な異常気象の影響
    df['multiple_extreme_conditions'] = (df['weather_stress'] >= 0.5).astype(int)
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure', 'pressure_local', 'sunshine_hours']
    
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
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

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去"""
    correlation_matrix = df[feature_columns].corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return [col for col in feature_columns if col not in to_drop]

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮した時系列分割を作成"""
    splits = []
    
    # データを年ごとに分割
    years = sorted(df['year'].unique())
    
    for i in range(len(years) - n_splits + 1):
        train_years = years[i:i+n_splits-1]
        test_year = years[i+n_splits-1]
        
        train_data = df[df['year'].isin(train_years)].copy()
        test_data = df[df['year'] == test_year].copy()
        
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        
        # 特徴量とターゲットを分離
        exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 'hospitalization_count', 'people']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 入院人数関連の特徴量を完全に除外
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
        
        # データの準備
        X_train = train_data[feature_columns].copy()
        y_train = train_data['target']
        X_test = test_data[feature_columns].copy()
        y_test = test_data['target']
        
        # NaN値の処理
        X_train = X_train.fillna(method='ffill').fillna(X_train.median())
        X_test = X_test.fillna(method='ffill').fillna(X_test.median())
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.median())
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        splits.append((X_train_scaled, y_train, X_test_scaled, y_test, feature_columns))
    
    return splits

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """アンサンブルモデルを学習"""
    models = {}
    
    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    models['lgb'] = lgb.LGBMClassifier(**lgb_params)
    models['lgb'].fit(X_train, y_train)
    
    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    models['xgb'] = xgb.XGBClassifier(**xgb_params)
    models['xgb'].fit(X_train, y_train)
    
    # CatBoost
    cb_params = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'Logloss',
        'random_state': 42,
        'verbose': False
    }
    models['catboost'] = cb.CatBoostClassifier(**cb_params)
    models['catboost'].fit(X_train, y_train)
    
    # 予測
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict_proba(X_val)[:, 1]
    
    # アンサンブル予測（平均）
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # AUC計算
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    
    return models, ensemble_pred, ensemble_auc

def main():
    """メイン実行関数（6分類天気概況版、動的選択）"""
    try:
        print("=== 6分類天気概況を含むAMI予測モデル（動的選択版） ===")
        
        # データの読み込み
        df = load_processed_data()
        
        # 日付でソート
        df = df.sort_values('hospitalization_date')
        
        print(f"全期間データ使用、データ数: {len(df)}")
        print("⚠️ 6分類天気概況を含む気象情報のみを使用したモデルです")
        print("⚠️ 元のモデルと同じ動的選択（30,50,70,100）を適用します")
        
        # ターゲット変数の作成（75%タイルで閾値を設定）
        threshold = df['hospitalization_count'].quantile(0.75)
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        print(f"ターゲット変数作成完了: 75%タイル閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
        
        # 天気概況を6分類に簡略化して数値化
        print("\n=== 天気概況の6分類簡略化 ===")
        df, weather_encoder = encode_simplified_weather_conditions(df)
        
        # 日付関連の特徴量を作成
        df = create_date_features(df)
        
        # 異常気象フラグを作成（元のモデルと同じ）
        df = detect_extreme_weather(df)
        
        # 気象要素の相互作用特徴量を作成
        df = create_weather_interaction_features(df)
        
        # 時系列特徴量を作成
        df = create_time_series_features(df)
        
        # 気象×時系列の高度な組み合わせ特徴量を作成
        df = create_advanced_weather_timeseries_features(df)
        
        # 季節性を考慮した重み付け特徴量を作成
        df = create_seasonal_weighted_features(df)
        
        # 特徴量の選択（使用しない列を除外）
        exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 'hospitalization_count', 'people']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 入院人数関連の特徴量を完全に除外
        hospitalization_related_cols = [
            col for col in feature_columns 
            if any(keyword in col.lower() for keyword in [
                'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
                'patients_std', 'patients_max', 'patients_min', 'dow_mean'
            ])
        ]
        feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
        
        print(f"使用特徴量数: {len(feature_columns)} (6分類天気概況を含む)")
        
        # 数値型の列のみを抽出
        numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
        feature_columns = list(numeric_cols)
        
        print(f"数値型特徴量数: {len(feature_columns)}")
        
        # 天気概況関連の特徴量を確認
        weather_condition_features = [col for col in feature_columns if 'weather' in col or 'encoded' in col]
        print(f"天気概況関連特徴量: {len(weather_condition_features)}個")
        for feature in weather_condition_features:
            print(f"  - {feature}")
        
        # 欠損値の処理
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 時系列分割の作成
        splits = create_seasonal_splits(df, n_splits=3)
        
        # 評価結果の保存用
        cv_results = []
        feature_importance = pd.DataFrame()
        
        # 各分割でモデルを学習・評価
        for fold, (X_train, y_train, X_test, y_test, feature_columns) in enumerate(splits, 1):
            print(f"\nFold {fold}の処理を開始します...")
            best_auc_fold = 0
            best_n = None
            best_features = feature_columns
            best_result = None
            auc_results = {}
            
            # 特徴量数を変えてループ（元のモデルと同じ）
            for n in [30, 50, 70, 100]:
                print(f"  特徴量数 {n} でテスト中...")
                
                # 1. LightGBMの特徴量重要度で上位n個を選択
                lgb_temp = lgb.LGBMClassifier(random_state=42)
                lgb_temp.fit(X_train, y_train)
                importances = lgb_temp.feature_importances_
                top_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])][:n]
                
                # 2. 高相関な特徴量を除去
                X_train_df = pd.DataFrame(X_train, columns=feature_columns)
                reduced_features = remove_highly_correlated_features(X_train_df, top_features)
                
                # 3. データを再構成
                X_train_sel = X_train_df[reduced_features].values
                X_test_sel = pd.DataFrame(X_test, columns=feature_columns)[reduced_features].values
                
                # 4. 標準化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_sel)
                X_test_scaled = scaler.transform(X_test_sel)
                
                # 5. アンサンブル学習
                try:
                    models, best_ensemble_pred, best_auc = train_ensemble_models(X_train_scaled, y_train, X_test_scaled, y_test)
                    auc_results[n] = best_auc
                    print(f"    特徴量数 {len(reduced_features)}: AUC = {best_auc:.4f}")
                    
                    # 天気概況関連の重要度を確認
                    weather_importance = [f for f in reduced_features if 'weather' in f or 'encoded' in f]
                    if weather_importance:
                        print(f"    選択された天気概況特徴量: {weather_importance}")
                    
                    if best_auc > best_auc_fold:
                        best_auc_fold = best_auc
                        best_n = n
                        best_features = reduced_features
                        best_result = (models, best_ensemble_pred, X_test_scaled)
                except Exception as e:
                    print(f"    エラー: {e}")
                    continue
            
            print(f"Fold {fold} 最良AUC: {best_auc_fold:.4f} (特徴量数: {len(best_features)})")
            
            if best_result is None:
                print(f"Fold {fold} で有効な結果が得られませんでした。スキップします。")
                continue
            
            # 最良の結果を使用
            models, best_ensemble_pred, X_test_scaled = best_result
            
            # 結果を保存
            cv_results.append({
                'fold': fold,
                'auc': best_auc_fold,
                'feature_count': len(best_features),
                'selected_features': best_features,
                'weather_features_included': len([f for f in best_features if 'weather' in f or 'encoded' in f]) > 0
            })
        
        # 全体の結果をまとめる
        mean_auc = np.mean([result['auc'] for result in cv_results])
        std_auc = np.std([result['auc'] for result in cv_results])
        
        print(f"\n=== 最終結果 ===")
        print(f"平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # 天気概況の選択状況をまとめる
        weather_included_folds = sum([result['weather_features_included'] for result in cv_results])
        total_folds = len(cv_results)
        
        print(f"\n=== 天気概況の選択状況 ===")
        print(f"天気概況が含まれたFold: {weather_included_folds}/{total_folds}")
        
        if weather_included_folds > 0:
            print("✅ 天気概況がモデルに貢献しました")
        else:
            print("⚠️ 天気概況は選択されませんでした（元のモデルと同じ性能）")
        
        # 選択された天気概況特徴量をまとめる
        all_selected_weather_features = []
        for result in cv_results:
            weather_features = [f for f in result['selected_features'] if 'weather' in f or 'encoded' in f]
            all_selected_weather_features.extend(weather_features)
        
        if all_selected_weather_features:
            unique_weather_features = list(set(all_selected_weather_features))
            print(f"\n=== 選択された天気概況特徴量 ===")
            for feature in unique_weather_features:
                count = all_selected_weather_features.count(feature)
                print(f"{feature:<35} - {count}回選択")
        
        # モデルを保存
        joblib.dump(models, 'weather_dynamic_models.pkl')
        joblib.dump(weather_encoder, 'weather_dynamic_encoder.pkl')
        
        print(f"\nモデルを保存しました: weather_dynamic_models.pkl")
        print(f"エンコーダーを保存しました: weather_dynamic_encoder.pkl")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 