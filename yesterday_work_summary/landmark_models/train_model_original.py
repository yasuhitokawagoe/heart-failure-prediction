import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
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

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week  # 週の情報を追加
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
    """異常気象フラグや気象ストレス指標を生成"""
    # NaN補完
    weather_cols = ['min_temp', 'max_temp', 'avg_temp', 'avg_wind', 'vapor_pressure', 
                   'avg_humidity', 'sunshine_hours', 'cloud_cover', 'pressure_local',
                   'snow_depth', 'snowfall_total', 'solar_radiation', 'precipitation', 
                   'precipitation_max_10min']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    # 熱帯夜
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    # 猛暑日
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    # 真夏日
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    # 冬日
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    # 真冬日
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    # 寒波（月平均-2℃以下）
    df['monthly_temp_mean'] = df.groupby(['year', 'month'])['avg_temp'].transform(lambda x: x.expanding().mean().shift(1).fillna(x.mean()))
    df['is_cold_wave'] = (df['avg_temp'] <= (df['monthly_temp_mean'] - 2.0)).astype(int)
    # 強風（95パーセンタイル超）
    df['wind_quantile'] = df['avg_wind'].expanding().quantile(0.95).shift(1).fillna(df['avg_wind'].quantile(0.95))
    df['is_strong_wind'] = (df['avg_wind'] > df['wind_quantile']).astype(int)
    # 台風（気圧低下＋強風）
    df['pressure_quantile'] = df['pressure_local'].expanding().quantile(0.1).shift(1).fillna(df['pressure_local'].quantile(0.1))
    df['wind_quantile_90'] = df['avg_wind'].expanding().quantile(0.9).shift(1).fillna(df['avg_wind'].quantile(0.9))
    df['is_typhoon_condition'] = ((df['pressure_local'] < df['pressure_quantile']) & (df['avg_wind'] > df['wind_quantile_90'])).astype(int)
    # 気圧ショック（前日比2σ超）
    df['pressure_change'] = df['pressure_local'] - df['pressure_local'].shift(1)
    df['pressure_change'] = df['pressure_change'].fillna(0)
    df['pressure_change_std'] = df['pressure_change'].expanding().std().shift(1).fillna(df['pressure_change'].std())
    df['is_rapid_pressure_change'] = (abs(df['pressure_change']) > df['pressure_change_std'] * 2).astype(int)
    # 高温多湿
    df['is_high_temp_humidity'] = ((df['avg_temp'] >= 28) & (df['avg_humidity'] >= 80)).astype(int)
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    # WBGT近似
    df['wbgt_approx'] = 0.7 * df['avg_humidity'] + 0.3 * df['avg_temp']
    # 風冷指数
    df['wind_chill'] = 13.12 + 0.6215 * df['avg_temp'] - 11.37 * (df['avg_wind'] ** 0.16) + 0.3965 * df['avg_temp'] * (df['avg_wind'] ** 0.16)
    # 気象ストレス合成指標
    df['weather_stress'] = (
        df['is_extremely_hot'] * 0.2 +
        df['is_cold_wave'] * 0.2 +
        df['is_rapid_pressure_change'] * 0.15 +
        df['is_strong_wind'] * 0.15 +
        df['is_typhoon_condition'] * 0.1 +
        df['is_high_temp_humidity'] * 0.1 +
        df['is_tropical_night'] * 0.05 +
        df['is_winter_day'] * 0.05
    )
    # 連続異常（3日連続）
    extreme_conditions = ['is_tropical_night', 'is_extremely_hot', 'is_cold_wave', 'is_strong_wind', 'is_typhoon_condition', 'is_high_temp_humidity']
    for condition in extreme_conditions:
        condition_shifted1 = df[condition].shift(1).fillna(0)
        condition_shifted2 = df[condition].shift(2).fillna(0)
        df[f'{condition}_consecutive'] = ((df[condition] == 1) & (condition_shifted1 == 1) & (condition_shifted2 == 1)).astype(int)
    return df

def create_weather_interaction_features(df):
    """気象要素のインタラクション・複合指標を生成"""
    # 気温×湿度、気温×気圧、気温×降水量、湿度×気圧など
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['pressure_local']
    df['temp_precipitation'] = df['avg_temp'] * df['precipitation']
    df['humidity_pressure'] = df['avg_humidity'] * df['pressure_local']
    df['humidity_precipitation'] = df['avg_humidity'] * df['precipitation']
    df['pressure_precipitation'] = df['pressure_local'] * df['precipitation']
    # 異常気象フラグ×気温・湿度・気圧
    for flag in ['is_tropical_night', 'is_extremely_hot', 'is_cold_wave', 'is_strong_wind', 'is_typhoon_condition', 'is_high_temp_humidity']:
        df[f'{flag}_temp'] = df[flag] * df['avg_temp']
        df[f'{flag}_humidity'] = df[flag] * df['avg_humidity']
        df[f'{flag}_pressure'] = df[flag] * df['pressure_local']
    return df

def create_time_series_features(df):
    """時系列特徴量を効率的に作成（データリーク防止・メモリ効率化・特徴量数削減）"""
    weather_cols = ['avg_temp', 'avg_humidity', 'pressure_local', 'precipitation', 
                   'snow_depth', 'cloud_cover', 'solar_radiation', 'avg_wind', 
                   'sunshine_hours', 'vapor_pressure']
    windows = [7, 30]  # 特徴量数を削減
    new_features = {}
    for col in weather_cols:
        for window in windows:
            # 移動平均
            new_features[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            # 移動標準偏差
            new_features[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
            # 移動最大値
            new_features[f'{col}_max_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).max()
            # 移動最小値
            new_features[f'{col}_min_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).min()
            # 移動中央値
            new_features[f'{col}_median_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).median()
            # 移動範囲
            new_features[f'{col}_range_{window}d'] = new_features[f'{col}_max_{window}d'] - new_features[f'{col}_min_{window}d']
    # DataFrameに一括追加
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    # 変化率・加速度も同様に効率化
    for col in weather_cols:
        df[f'{col}_change_rate'] = (df[col] - df[col].shift(1)) / df[col].shift(1)
        df[f'{col}_weekly_change_rate'] = (df[col] - df[col].shift(7)) / df[col].shift(7)
        df[f'{col}_monthly_change_rate'] = (df[col] - df[col].shift(30)) / df[col].shift(30)
        df[f'{col}_acceleration'] = df[f'{col}_change_rate'] - df[f'{col}_change_rate'].shift(1)
    return df

def load_processed_data():
    """ラグなしデータセットを読み込み、ラグ特徴量を自動生成"""
    import pandas as pd
    import numpy as np
    try:
        # 絶対パスでファイルを指定
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        print(f"ラグなしデータ読み込み完了: {len(df)}行, {len(df.columns)}カラム")
        # 日付カラムの変換
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        # カラム名の統一
        if 'people' in df.columns:
            df['hospitalization_count'] = df['people']
        # 数値型カラムのリスト
        numeric_columns = [
            'avg_temp', 'max_temp', 'min_temp', 'sunshine_hours', 'avg_wind', 'avg_humidity',
            'cloud_cover', 'pressure_local', 'vapor_pressure', 'snow_depth', 'snowfall_total',
            'solar_radiation', 'precipitation', 'precipitation_max_10min'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"データ前処理完了: 数値カラム {len(numeric_columns)}個")
        return df
    except FileNotFoundError:
        print("ラグなしデータファイルが見つかりません。")
        raise

def add_lag_features(df, lag_days=7):
    """主要な気象変数・天気分類などに1〜lag_days日ラグ特徴量を追加"""
    # ラグを作成するカラム
    lag_columns = [
        'avg_temp', 'max_temp', 'min_temp', 'sunshine_hours', 'avg_wind', 'avg_humidity',
        'cloud_cover', 'pressure_local', 'vapor_pressure', 'snow_depth', 'snowfall_total',
        'solar_radiation', 'precipitation', 'precipitation_max_10min',
        '天気概況(昼：06時～18時)', '天気概況(夜：18時～翌日06時)',
        '天気分類(昼)', '天気分類(夜)', '天気分類(統合)'
    ]
    for col in lag_columns:
        if col in df.columns:
            for lag in range(1, lag_days+1):
                df[f'{col}_{lag}day_ago'] = df[col].shift(lag)
    return df

def create_features_for_date(df, current_date):
    """特定の日付のデータに対して特徴量を作成（前日までのデータのみを使用）"""
    # 前日までのデータを抽出
    historical_data = df[df['hospitalization_date'] < current_date].copy()
    current_data = df[df['hospitalization_date'] == current_date].copy()
    
    if len(historical_data) == 0 or len(current_data) == 0:
        return None
    
    # 日付関連の特徴量
    current_data['year'] = current_date.year
    current_data['month'] = current_date.month
    current_data['day'] = current_date.day
    current_data['dayofweek'] = current_date.dayofweek
    current_data['is_weekend'] = current_date.dayofweek in [5, 6]
    current_data['is_holiday'] = int(jpholiday.is_holiday(current_date) or current_date.weekday() in [5, 6])
    
    # 季節性指標（数値として）
    month = current_date.month
    if month in [12, 1, 2]:
        current_data['season_winter'] = 1
        current_data['season_spring'] = 0
        current_data['season_summer'] = 0
        current_data['season_autumn'] = 0
    elif month in [3, 4, 5]:
        current_data['season_winter'] = 0
        current_data['season_spring'] = 1
        current_data['season_summer'] = 0
        current_data['season_autumn'] = 0
    elif month in [6, 7, 8]:
        current_data['season_winter'] = 0
        current_data['season_spring'] = 0
        current_data['season_summer'] = 1
        current_data['season_autumn'] = 0
    else:  # 9, 10, 11
        current_data['season_winter'] = 0
        current_data['season_spring'] = 0
        current_data['season_summer'] = 0
        current_data['season_autumn'] = 1
    
    # 過去の入院数に基づく特徴量（前日までのデータのみを使用）
    for lag in [1, 2, 3, 7, 14, 28]:
        if len(historical_data) >= lag:
            current_data[f'patients_lag_{lag}'] = historical_data['hospitalization_count'].iloc[-lag]
        else:
            current_data[f'patients_lag_{lag}'] = np.nan
    
    # 移動平均特徴量（前日までのデータのみを使用）
    for window in [7, 14, 28]:
        if len(historical_data) >= window:
            current_data[f'patients_ma_{window}'] = historical_data['hospitalization_count'].tail(window).mean()
            current_data[f'patients_std_{window}'] = historical_data['hospitalization_count'].tail(window).std()
            current_data[f'patients_max_{window}'] = historical_data['hospitalization_count'].tail(window).max()
            current_data[f'patients_min_{window}'] = historical_data['hospitalization_count'].tail(window).min()
        else:
            current_data[f'patients_ma_{window}'] = np.nan
            current_data[f'patients_std_{window}'] = np.nan
            current_data[f'patients_max_{window}'] = np.nan
            current_data[f'patients_min_{window}'] = np.nan
    
    # 曜日ごとの統計量（前日までのデータのみを使用）
    if len(historical_data) >= 28:
        dow_data = historical_data[historical_data['hospitalization_date'].dt.dayofweek == current_date.dayofweek]
        current_data['dow_mean_28d'] = dow_data['hospitalization_count'].tail(4).mean()  # 過去4回の同じ曜日の平均
    else:
        current_data['dow_mean_28d'] = np.nan
    
    # 気象要素の特徴量（前日までのデータのみを使用）
    if len(historical_data) >= 1:
        yesterday_data = historical_data.iloc[-1]
        current_data['temp_change'] = current_data['avg_temp'] - yesterday_data['avg_temp']
        current_data['humidity_change'] = current_data['avg_humidity'] - yesterday_data['avg_humidity']
        current_data['pressure_change'] = current_data['pressure_local'] - yesterday_data['pressure_local']
    else:
        current_data['temp_change'] = np.nan
        current_data['humidity_change'] = np.nan
        current_data['pressure_change'] = np.nan
    
    return current_data

def prepare_data_for_training(df, train_dates, test_dates):
    """トレーニングデータとテストデータを準備"""
    # データを日付で分割
    train_data = df[df['hospitalization_date'].isin(train_dates)].copy()
    test_data = df[df['hospitalization_date'].isin(test_dates)].copy()
    
    # 特徴量とターゲットを分離
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院データに関連する特徴量を除外（当日のデータのみ）
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people'  # 当日のデータのみを除外
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

    # 追加: inf→NaN→中央値補完
    train_data[feature_columns] = train_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
    test_data[feature_columns] = test_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
    
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
        
        # データの準備
        X_train, y_train, X_test, y_test, feature_columns = prepare_data_for_training(df, train_dates, test_dates)
        
        splits.append((X_train, y_train, X_test, y_test, feature_columns))
        
        print(f"Split {i+1}:")
        print(f"訓練期間: {train_dates.min()} から {train_dates.max()}")
        print(f"テスト期間: {test_dates.min()} から {test_dates.max()}\n")
    
    return splits

def optimize_lgb_params(X_train, y_train, X_val, y_val):
    """LightGBMのハイパーパラメータを最適化"""
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100)
        }
        
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    
    return study.best_params

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """複数のモデルを学習し、アンサンブルモデルを作成"""
    # NaN値の処理
    X_train = np.nan_to_num(X_train, nan=0)
    X_val = np.nan_to_num(X_val, nan=0)
    
    models = {}
    predictions = {}
    # LightGBM
    print("Training LightGBM...")
    lgb_params = optimize_lgb_params(X_train, y_train, X_val, y_val)
    train_data = lgb.Dataset(X_train, y_train)
    models['lgb'] = lgb.train({**lgb_params, 'verbose': -1}, train_data)
    predictions['lgb'] = models['lgb'].predict(X_val)
    # XGBoost
    print("Training XGBoost...")
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
    }
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    models['xgb'] = xgb.train(xgb_params, dtrain, num_boost_round=1000,
                             early_stopping_rounds=50, evals=[(dval, 'val')],
                             verbose_eval=False)
    predictions['xgb'] = models['xgb'].predict(xgb.DMatrix(X_val))
    # CatBoost
    print("Training CatBoost...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),
        'eval_metric': 'AUC',
        'verbose': False
    }
    models['cat'] = cb.CatBoostClassifier(**cat_params)
    models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val),
                     early_stopping_rounds=50, verbose=False)
    predictions['cat'] = models['cat'].predict_proba(X_val)[:, 1]
    # Neural Network
    print("Training Neural Network...")
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    models['nn'] = nn_model
    models['nn'].fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=32, verbose=0)
    predictions['nn'] = models['nn'].predict(X_val).flatten()
    # アンサンブルの重みを最適化
    def optimize_weights(weights):
        weighted_pred = np.zeros_like(predictions['lgb'])
        for model_name, weight in zip(['lgb', 'xgb', 'cat', 'nn'], weights):
            weighted_pred += weight * predictions[model_name]
        # NaN値を処理
        weighted_pred = np.nan_to_num(weighted_pred, nan=0.5)
        return -roc_auc_score(y_val, weighted_pred)
    # 制約条件：重みの合計が1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(0, 1)] * 4
    initial_weights = [0.25] * 4
    result = minimize(optimize_weights, initial_weights,
                     method='SLSQP', bounds=bounds,
                     constraints=constraints)
    optimal_weights = result.x
    print("\nOptimal ensemble weights:")
    for model_name, weight in zip(['LightGBM', 'XGBoost', 'CatBoost', 'Neural Network'],
                                optimal_weights):
        print(f"{model_name}: {weight:.3f}")
    return models, optimal_weights

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

def plot_model_performance(y_true, y_pred_proba, fold_num=None):
    """モデルの性能を可視化する関数"""
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(12, 5))
    # ROC曲線のプロット
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    # PR曲線のプロット
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    # 保存するファイル名にfold番号を含める
    fold_str = f'_fold_{fold_num}' if fold_num is not None else ''
    plt.savefig(f'results/performance_curves{fold_str}.png')
    plt.close()

def save_detailed_results(cv_results, feature_importance=None):
    """詳細な結果を保存する関数"""
    os.makedirs('results', exist_ok=True)
    # 全体の評価指標
    overall_metrics = {
        'roc_auc': {
            'mean': np.mean([r['roc_auc'] for r in cv_results]),
            'std': np.std([r['roc_auc'] for r in cv_results])
        },
        'pr_auc': {
            'mean': np.mean([r['pr_auc'] for r in cv_results]),
            'std': np.std([r['pr_auc'] for r in cv_results])
        },
        'precision': {
            'mean': np.mean([r['precision'] for r in cv_results]),
            'std': np.std([r['precision'] for r in cv_results])
        },
        'recall': {
            'mean': np.mean([r['recall'] for r in cv_results]),
            'std': np.std([r['recall'] for r in cv_results])
        },
        'f1_score': {
            'mean': np.mean([r['f1_score'] for r in cv_results]),
            'std': np.std([r['f1_score'] for r in cv_results])
        },
        'specificity': {
            'mean': np.mean([r['specificity'] for r in cv_results]),
            'std': np.std([r['specificity'] for r in cv_results])
        }
    }
    # 結果をJSONファイルとして保存
    with open('results/detailed_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    # Fold別の結果をMarkdownファイルとして保存
    with open('results/fold_results.md', 'w') as f:
        f.write('# モデル評価結果\n\n')
        # 全体の結果
        f.write('## 全体の性能\n\n')
        f.write('| 指標 | 平均 ± 標準偏差 |\n')
        f.write('|------|----------------|\n')
        for metric, values in overall_metrics.items():
            f.write(f"| {metric} | {values['mean']:.4f} ± {values['std']:.4f} |\n")
        # Fold別の結果
        f.write('\n## Fold別の性能\n\n')
        for i, result in enumerate(cv_results, 1):
            f.write(f'\n### Fold {i}\n\n')
            f.write('| 指標 | 値 |\n')
            f.write('|------|----|\n')
            for metric, value in result.items():
                if metric != 'confusion_matrix':
                    f.write(f'| {metric} | {value:.4f} |\n')
            f.write('\n#### 混同行列\n\n')
            f.write('|              | 予測: 通常 | 予測: 多い |\n')
            f.write('|--------------|------------|------------|\n')
            cm = result['confusion_matrix']
            f.write(f'| 実際: 通常 | {cm[0,0]} | {cm[0,1]} |\n')
            f.write(f'| 実際: 多い | {cm[1,0]} | {cm[1,1]} |\n')
    # 特徴量の重要度をプロット（存在する場合）
    if feature_importance is not None:
        plt.figure(figsize=(12, 6))
        feature_importance.sort_values('importance', ascending=True).plot(kind='barh', x='feature', y='importance')
        plt.title('特徴量の重要度')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png')
        plt.close()
        # 追加: 重要度をCSVでも保存
        feature_importance.to_csv('results/feature_importance.csv', index=False)

def save_modelwise_metrics(modelwise_metrics):
    os.makedirs('results', exist_ok=True)
    # JSON保存
    with open('results/modelwise_metrics.json', 'w') as f:
        json.dump(modelwise_metrics, f, indent=4, ensure_ascii=False)
    # Markdown保存
    with open('results/modelwise_metrics.md', 'w') as f:
        f.write('# 各モデルごとの評価指標\n\n')
        f.write('| Fold | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Specificity |\n')
        f.write('|------|-------|---------|--------|-----------|--------|----|-------------|\n')
        for m in modelwise_metrics:
            f.write(f"| {m['fold']} | {m['model']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['specificity']:.4f} |\n")

def main():
    """メイン実行関数"""
    try:
        # データの読み込み
        df = load_processed_data()
        df = add_lag_features(df, lag_days=7)
        
        # 日付でソート
        df = df.sort_values('hospitalization_date')
        
        # 期間制限を解除（2019-12-31までの制限を削除）
        # df = df[df['hospitalization_date'] <= '2019-12-31'].copy()
        # print(f"データ期間制限: 2012-2019年、データ数: {len(df)}")
        
        # ターゲット変数の作成（75%タイルで閾値を設定）
        threshold = df['hospitalization_count'].quantile(0.75)  # 75%タイル
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        print(f"ターゲット変数作成完了: 75%タイル閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
        print("hospitalization_count describe:")
        print(df['hospitalization_count'].describe())
        print("target value_counts:")
        print(df['target'].value_counts())
        print("hospitalization_countサンプル:")
        print(df['hospitalization_count'].head(10))
        
        # 日付関連の特徴量を作成
        df = create_date_features(df)
        
        # 元のモデル用：新規追加分は除外
        # 異常気象フラグや気象ストレス指標を生成
        # df = detect_extreme_weather(df)
        
        # 気象要素のインタラクション・複合指標を生成
        # df = create_weather_interaction_features(df)
        
        # 時系列特徴量を作成
        # df = create_time_series_features(df)
        
        # 特徴量の選択（使用しない列を除外）
        exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date']
        feature_columns = [col for col in df.columns if col not in exclude_cols]

        # 数値型の列のみを抽出
        numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns

        # 追加: inf→NaN→中央値補完
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        # 最終的なNaNチェック
        if df[numeric_cols].isnull().any().any():
            print("Warning: NaN values still exist after cleaning")
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # 時系列分割の作成
        splits = create_seasonal_splits(df, n_splits=3)
        
        # 評価結果の保存用
        cv_results = []
        feature_importance = pd.DataFrame()
        # 各モデルごとの評価指標を保存するリスト
        modelwise_metrics = []
        
        # 各分割でモデルを学習・評価
        for fold, (X_train, y_train, X_test, y_test, feature_columns) in enumerate(splits, 1):
            print(f"\nFold {fold}の処理を開始します...")
            
            # 特徴量とターゲットの準備
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # アンサンブルモデルの学習
            print("アンサンブルモデルを学習しています...")
            models, weights = train_ensemble_models(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # テストデータでの予測
            predictions = {}
            for model_name, model in models.items():
                if model_name == 'lgb':
                    predictions[model_name] = model.predict(X_test_scaled)
                elif model_name == 'xgb':
                    predictions[model_name] = model.predict(xgb.DMatrix(X_test_scaled))
                elif model_name == 'cat':
                    predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]
                else:  # Neural Network
                    predictions[model_name] = model.predict(X_test_scaled).flatten()
            
            # 各モデルごとの評価指標を計算
            for model_name, y_pred_proba in predictions.items():
                # 追加: 予測値の分布をprint
                print(f"{model_name}予測値: min={np.min(y_pred_proba):.3f}, max={np.max(y_pred_proba):.3f}, mean={np.mean(y_pred_proba):.3f}, std={np.std(y_pred_proba):.3f}")
                print(f"{model_name}予測値サンプル: {y_pred_proba[:10]}")
                # 追加: y_pred_probaにNaNがあれば中央値で補完
                if np.isnan(y_pred_proba).any():
                    print(f"{model_name}の予測値にNaNが含まれていたため中央値で補完します")
                    if np.isnan(y_pred_proba).all():
                        print(f"{model_name}の予測値が全てNaNのため0.5で補完します")
                        y_pred_proba = np.full_like(y_pred_proba, 0.5)
                    else:
                        median_pred = np.nanmedian(y_pred_proba)
                        y_pred_proba = np.where(np.isnan(y_pred_proba), median_pred, y_pred_proba)
                # evaluate_model_performance直前でもNaNが残っていれば0.5で補完
                if np.isnan(y_pred_proba).any():
                    print(f"{model_name}の予測値にNaNが残っていたため0.5で補完します")
                    y_pred_proba = np.where(np.isnan(y_pred_proba), 0.5, y_pred_proba)
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
            ensemble_pred = np.zeros_like(predictions['lgb'])
            for model_name, weight in zip(['lgb', 'xgb', 'cat', 'nn'], weights):
                ensemble_pred += weight * predictions[model_name]
            # 追加: アンサンブル予測値の分布をprint
            print(f"ensemble_pred: min={np.min(ensemble_pred):.3f}, max={np.max(ensemble_pred):.3f}, mean={np.mean(ensemble_pred):.3f}, std={np.std(ensemble_pred):.3f}")
            print(f"ensemble_predサンプル: {ensemble_pred[:10]}")
            # 追加: ensemble_predにNaNがあれば中央値で補完
            if np.isnan(ensemble_pred).any():
                print("アンサンブル予測値にNaNが含まれていたため中央値で補完します")
                if np.isnan(ensemble_pred).all():
                    print("アンサンブル予測値が全てNaNのため0.5で補完します")
                    ensemble_pred = np.full_like(ensemble_pred, 0.5)
                else:
                    median_pred = np.nanmedian(ensemble_pred)
                    ensemble_pred = np.where(np.isnan(ensemble_pred), median_pred, ensemble_pred)
            # evaluate_model_performance直前でもNaNが残っていれば0.5で補完
            if np.isnan(ensemble_pred).any():
                print("アンサンブル予測値にNaNが残っていたため0.5で補完します")
                ensemble_pred = np.where(np.isnan(ensemble_pred), 0.5, ensemble_pred)
            # モデルの評価
            fold_metrics = evaluate_model_performance(y_test, ensemble_pred)
            cv_results.append(fold_metrics)
            
            # 性能曲線のプロット
            plot_model_performance(y_test, ensemble_pred, fold)
            
            # 特徴量の重要度（LightGBMの結果を使用）
            if fold == 1:  # 最初のfoldの特徴量重要度のみを保存
                importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': models['lgb'].feature_importance()
                })
                importance = importance.sort_values('importance', ascending=False)
                feature_importance = importance
        
        # 詳細な結果の保存
        save_detailed_results(cv_results, feature_importance)
        # 各モデルごとの評価指標を保存
        save_modelwise_metrics(modelwise_metrics)
        
        # 最終的な性能の表示
        print("\n=== 最終的な評価結果 ===")
        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score', 'specificity']
        for metric in metrics:
            values = [r[metric] for r in cv_results]
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 