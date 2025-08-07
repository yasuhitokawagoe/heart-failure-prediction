# 東京全体の入院リスクを気象情報のみで予測するモデル
# ベース: yesterday_work_summary/train_model_weather_only.py
# 必要な変数名・データパスのみ東京全体用に修正

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

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week  # 週の情報を追加
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 祝日フラグ
    df['is_holiday'] = df['date'].apply(
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
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # 四半期
    df['quarter'] = df['date'].dt.quarter
    
    return df

def detect_extreme_weather(df):
    """異常気象を検出する関数"""
    # NaN値を前方補完（未来のデータは使用しない）
    weather_cols = ['min_temp_weather', 'max_temp_weather', 'avg_temp_weather', 'avg_wind_weather', 'pressure_local', 
                   'avg_humidity_weather', 'sunshine_hours_weather']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    
    # 残りのNaN値を中央値で補完
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 熱帯夜（夜間の最低気温が25℃以上）
    df['is_tropical_night'] = (df['min_temp_weather'] >= 25).astype(int)
    
    # 猛暑日（最高気温が35℃以上）
    df['is_extremely_hot'] = (df['max_temp_weather'] >= 35).astype(int)
    
    # 真夏日（最高気温が30℃以上）
    df['is_hot_day'] = (df['max_temp_weather'] >= 30).astype(int)
    
    # 夏日（最高気温が25℃以上）
    df['is_summer_day'] = (df['max_temp_weather'] >= 25).astype(int)
    
    # 冬日（最低気温が0℃未満）
    df['is_winter_day'] = (df['min_temp_weather'] < 0).astype(int)
    
    # 真冬日（最高気温が0℃未満）
    df['is_freezing_day'] = (df['max_temp_weather'] < 0).astype(int)
    
    # 寒波（その月の平均気温から大きく低い）
    df['monthly_temp_mean'] = df.groupby(['year', 'month'])['avg_temp_weather'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )
    df['is_cold_wave'] = (df['avg_temp_weather'] <= (df['monthly_temp_mean'] - 2.0)).astype(int)
    
    # 強風（平均風速が高い）
    df['wind_quantile'] = df['avg_wind_weather'].expanding().quantile(0.95).shift(1).fillna(df['avg_wind_weather'].quantile(0.95))
    df['is_strong_wind'] = (df['avg_wind_weather'] > df['wind_quantile']).astype(int)
    
    # 台風の可能性（気圧が低く、風速が強い状態）
    df['pressure_quantile'] = df['pressure_local'].expanding().quantile(0.1).shift(1).fillna(df['pressure_local'].quantile(0.1))
    df['wind_quantile_90'] = df['avg_wind_weather'].expanding().quantile(0.9).shift(1).fillna(df['avg_wind_weather'].quantile(0.9))
    df['is_typhoon_condition'] = ((df['pressure_local'] < df['pressure_quantile']) & 
                                 (df['avg_wind_weather'] > df['wind_quantile_90'])).astype(int)
    
    # 気圧の急激な変化（3日間で10hPa以上の変化）
    df['pressure_change_3d'] = df['pressure_local'] - df['pressure_local'].shift(3)
    df['is_rapid_pressure_change'] = (abs(df['pressure_change_3d']) > 10).astype(int)
    
    # 気温の急激な変化（3日間で5℃以上の変化）
    df['temp_change_3d'] = df['avg_temp_weather'] - df['avg_temp_weather'].shift(3)
    df['is_rapid_temp_change'] = (abs(df['temp_change_3d']) > 5).astype(int)
    
    # 気温の変動性（7日間の標準偏差）
    df['temp_volatility'] = df['avg_temp_weather'].rolling(window=7, min_periods=1).std().shift(1).fillna(0)
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
    df['temp_quantile_80'] = df['avg_temp_weather'].expanding().quantile(0.8).shift(1).fillna(df['avg_temp_weather'].quantile(0.8))
    df['humidity_quantile_80'] = df['avg_humidity_weather'].expanding().quantile(0.8).shift(1).fillna(df['avg_humidity_weather'].quantile(0.8))
    df['temp_humidity_stress'] = (
        (df['avg_temp_weather'] > df['temp_quantile_80']) & 
        (df['avg_humidity_weather'] > df['humidity_quantile_80'])
    ).astype(int)
    
    # 気温と風速の複合的な影響
    df['wind_quantile_80'] = df['avg_wind_weather'].expanding().quantile(0.8).shift(1).fillna(df['avg_wind_weather'].quantile(0.8))
    df['temp_wind_stress'] = (
        (df['avg_temp_weather'] > df['temp_quantile_80']) & 
        (df['avg_wind_weather'] > df['wind_quantile_80'])
    ).astype(int)
    
    # 気圧と風速の複合的な影響
    df['pressure_quantile_20'] = df['pressure_local'].expanding().quantile(0.2).shift(1).fillna(df['pressure_local'].quantile(0.2))
    df['pressure_wind_stress'] = (
        (df['pressure_local'] < df['pressure_quantile_20']) & 
        (df['avg_wind_weather'] > df['wind_quantile_80'])
    ).astype(int)
    
    # 日照時間と気温の複合的な影響
    df['sunshine_quantile_80'] = df['sunshine_hours_weather'].expanding().quantile(0.8).shift(1).fillna(df['sunshine_hours_weather'].quantile(0.8))
    df['sunshine_temp_stress'] = (
        (df['sunshine_hours_weather'] > df['sunshine_quantile_80']) & 
        (df['avg_temp_weather'] > df['temp_quantile_80'])
    ).astype(int)
    
    # 地域特有の気象パターン（東京特有の気象条件）
    df['tokyo_specific_heat'] = (
        (df['month'].isin([7, 8])) & 
        (df['avg_temp_weather'] > df['temp_quantile_80']) & 
        (df['avg_humidity_weather'] > df['humidity_quantile_80'])
    ).astype(int)
    
    # temp_quantile_20の定義を追加
    df['temp_quantile_20'] = df['avg_temp_weather'].expanding().quantile(0.2).shift(1).fillna(df['avg_temp_weather'].quantile(0.2))
    
    df['tokyo_specific_cold'] = (
        (df['month'].isin([1, 2])) & 
        (df['avg_temp_weather'] < df['temp_quantile_20']) & 
        (df['avg_wind_weather'] > df['wind_quantile_80'])
    ).astype(int)
    
    # より複雑な気象要素の組み合わせ
    df['complex_weather_stress'] = (
        df['temp_humidity_stress'] * 0.3 +
        df['temp_wind_stress'] * 0.3 +
        df['pressure_wind_stress'] * 0.2 +
        df['sunshine_temp_stress'] * 0.2
    )
    
    # 気象要素の非線形な相互作用
    df['nonlinear_temp_humidity'] = np.exp(-0.1 * (df['avg_temp_weather'] - df['avg_humidity_weather'])**2)
    df['nonlinear_temp_pressure'] = np.exp(-0.1 * (df['avg_temp_weather'] - df['pressure_local'])**2)
    
    # 気象要素の急激な変化の検出（加速度）
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        df[f'{col}_acceleration'] = (
            df[col] - 2 * df[col].shift(1) + df[col].shift(2)
        ).fillna(0)
        df[f'{col}_std'] = df[f'{col}_acceleration'].expanding().std().shift(1).fillna(
            df[f'{col}_acceleration'].std()
        )
        df[f'{col}_rapid_change'] = (abs(df[f'{col}_acceleration']) > 
                                   df[f'{col}_std'] * 2).astype(int)
    
    return df

def load_processed_data():
    """処理済みデータを読み込み（東京全体用）"""
    try:
        # 東京全体_analysisディレクトリの統合データを使用
        df = pd.read_csv('tokyo_weather_merged.csv')
        df['date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people_tokyo']  # 東京全体の入院件数
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_features_for_date(df, current_date):
    """特定の日付のデータに対して特徴量を作成（気象情報のみ）"""
    # 前日までのデータを抽出
    historical_data = df[df['date'] < current_date].copy()
    current_data = df[df['date'] == current_date].copy()
    
    if len(historical_data) == 0 or len(current_data) == 0:
        return None
    
    # 日付関連の特徴量
    current_data['year'] = current_date.year
    current_data['month'] = current_date.month
    current_data['day'] = current_date.day
    current_data['dayofweek'] = current_date.dayofweek
    current_data['is_weekend'] = current_date.dayofweek in [5, 6]
    current_data['is_holiday'] = int(jpholiday.is_holiday(current_date) or current_date.weekday() in [5, 6])
    
    # 入院人数関連の特徴量は完全に除外（気象情報のみ使用）
    
    # 気象要素の特徴量（前日までのデータのみを使用）
    if len(historical_data) >= 1:
        yesterday_data = historical_data.iloc[-1]
        current_data['temp_change'] = current_data['avg_temp_weather'] - yesterday_data['avg_temp_weather']
        current_data['humidity_change'] = current_data['avg_humidity_weather'] - yesterday_data['avg_humidity_weather']
        current_data['pressure_change'] = current_data['pressure_local'] - yesterday_data['pressure_local']
    else:
        current_data['temp_change'] = np.nan
        current_data['humidity_change'] = np.nan
        current_data['pressure_change'] = np.nan
    
    return current_data

def prepare_data_for_training(df, train_dates, test_dates):
    """トレーニングデータとテストデータを準備（気象情報のみ）"""
    # データを日付で分割
    train_data = df[df['date'].isin(train_dates)].copy()
    test_data = df[df['date'].isin(test_dates)].copy()
    
    # 特徴量とターゲットを分離
    exclude_cols = ['date', 'target', 'season', 'prefecture_name', 'hospitalization_count', 'people_tokyo', 'people_weather']
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

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮した時系列分割を作成"""
    splits = []
    unique_dates = pd.Series(df['date'].unique()).sort_values()
    total_days = len(unique_dates)
    
    # 各分割のサイズを計算
    split_size = total_days // (n_splits + 1)
    
    for i in range(n_splits):
        # 訓練期間の終了日
        train_end_idx = (i + 1) * split_size
        # テスト期間の終了日
        test_end_idx = train_end_idx + split_size
        
        # 訓練期間とテスト期間の日付を取得
        train_dates = unique_dates[:train_end_idx]
        test_dates = unique_dates[train_end_idx:test_end_idx]
        
        splits.append((train_dates, test_dates))
    
    return splits

# --- 以降の関数はAMI用コードをそのままコピーし、
# 必要に応じて保存先パスなどを東京全体_analysis用に修正 ---

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

def create_deep_nn_model(input_dim):
    """より深いNeural Networkモデルを作成（強化版）"""
    model = tf.keras.Sequential([
        # 入力層
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # 第1隠れ層
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # 第2隠れ層
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # 第3隠れ層
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        
        # 出力層
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 学習率スケジューラー
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )
    return model

def create_attention_nn_model(input_dim):
    """Attention機構付きNeural Networkモデルを作成"""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # 特徴量の重要度を学習するAttention層
    attention_weights = tf.keras.layers.Dense(input_dim, activation='softmax')(inputs)
    attended_features = tf.keras.layers.Multiply()([inputs, attention_weights])
    
    # 深いネットワーク
    x = tf.keras.layers.Dense(256, activation='relu')(attended_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 学習率スケジューラー
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )
    return model

def create_ensemble_weights_optimizer():
    """アンサンブル重みの動的最適化器を作成"""
    def objective(weights, predictions, y_true):
        # 重みを正規化
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 重み付きアンサンブル予測
        ensemble_pred = np.zeros(len(y_true))
        for i, (model_name, pred) in enumerate(predictions.items()):
            ensemble_pred += weights[i] * pred
        
        # AUCを最大化
        return -roc_auc_score(y_true, ensemble_pred)
    
    return objective

def optimize_ensemble_weights_dynamic(predictions, y_true, method='scipy'):
    """アンサンブル重みを動的に最適化"""
    n_models = len(predictions)
    
    if method == 'scipy':
        # scipy.optimizeを使用
        objective = create_ensemble_weights_optimizer()
        
        # 初期重み（均等）
        initial_weights = np.ones(n_models) / n_models
        
        # 制約条件（重みの合計が1）
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # 境界条件（重みが0以上）
        bounds = [(0, 1) for _ in range(n_models)]
        
        # 最適化
        result = minimize(
            lambda w: objective(w, predictions, y_true),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # 正規化
            return optimal_weights
        else:
            print("Warning: Optimization failed, using equal weights")
            return np.ones(n_models) / n_models
    
    elif method == 'grid_search':
        # グリッドサーチ（小規模な場合）
        best_auc = 0
        best_weights = np.ones(n_models) / n_models
        
        # 重みの組み合わせを試す
        weight_steps = np.linspace(0, 1, 11)
        for weights in itertools.product(weight_steps, repeat=n_models):
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
                ensemble_pred = np.zeros(len(y_true))
                for i, (model_name, pred) in enumerate(predictions.items()):
                    ensemble_pred += weights[i] * pred
                
                auc_score = roc_auc_score(y_true, ensemble_pred)
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_weights = weights
        
        return best_weights
    
    else:
        # デフォルトは均等重み
        return np.ones(n_models) / n_models

def calculate_model_correlation(predictions):
    """モデル間の相関を計算"""
    pred_df = pd.DataFrame(predictions)
    return pred_df.corr()

def create_diversified_ensemble(predictions, y_true):
    """多様性を考慮したアンサンブルを作成"""
    # モデル間の相関を計算
    correlation_matrix = calculate_model_correlation(predictions)
    
    # 相関が低いモデルの組み合わせを選択
    n_models = len(predictions)
    model_names = list(predictions.keys())
    
    # 各モデルの性能を評価
    model_performances = {}
    for model_name, pred in predictions.items():
        auc_score = roc_auc_score(y_true, pred)
        model_performances[model_name] = auc_score
    
    # 性能と多様性を考慮した重み付け
    # 性能が高く、他のモデルとの相関が低いモデルにより高い重みを与える
    weights = np.ones(n_models)
    
    for i, model_name in enumerate(model_names):
        # 性能スコア
        performance_score = model_performances[model_name]
        
        # 多様性スコア（他のモデルとの平均相関の逆数）
        correlations = correlation_matrix.iloc[i].drop(model_name)
        diversity_score = 1 / (1 + correlations.mean())
        
        # 総合スコア
        weights[i] = performance_score * diversity_score
    
    # 正規化
    weights = weights / np.sum(weights)
    
    return weights

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """アンサンブルモデルを訓練"""
    models = {}
    predictions = {}
    
    # 1. LightGBM
    print("Training LightGBM...")
    lgb_params = optimize_lgb_params(X_train, y_train, X_val, y_val)
    lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    predictions['lgb'] = lgb_model.predict_proba(X_val)[:, 1]
    
    # 2. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_model.predict_proba(X_val)[:, 1]
    
    # 3. CatBoost
    print("Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    models['cat'] = cat_model
    predictions['cat'] = cat_model.predict_proba(X_val)[:, 1]
    
    # 4. Deep Neural Network
    print("Training Deep Neural Network...")
    try:
        nn_model = create_deep_nn_model(X_train.shape[1])
        nn_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        models['nn'] = nn_model
        predictions['nn'] = nn_model.predict(X_val).flatten()
    except Exception as e:
        print(f"Neural Network training failed: {e}")
    
    # 5. Attention Neural Network
    print("Training Attention Neural Network...")
    try:
        attention_model = create_attention_nn_model(X_train.shape[1])
        attention_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        models['attention_nn'] = attention_model
        predictions['attention_nn'] = attention_model.predict(X_val).flatten()
    except Exception as e:
        print(f"Attention Neural Network training failed: {e}")
    
    # アンサンブル重みの最適化
    print("Optimizing ensemble weights...")
    optimal_weights = optimize_ensemble_weights_dynamic(predictions, y_val)
    
    # 最適なアンサンブル予測
    ensemble_pred = np.zeros(len(y_val))
    for i, (model_name, pred) in enumerate(predictions.items()):
        ensemble_pred += optimal_weights[i] * pred
    
    print(f"Optimal weights: {dict(zip(predictions.keys(), optimal_weights))}")
    
    return models, predictions, ensemble_pred, optimal_weights

def perform_stacking(models, X_train, y_train, X_val):
    """Stacking（メタ学習）を実行"""
    # レベル0の予測を取得
    level0_predictions = {}
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            level0_predictions[model_name] = model.predict_proba(X_val)[:, 1]
        else:
            level0_predictions[model_name] = model.predict(X_val).flatten()
    
    # レベル1のメタ学習器（LightGBM）
    meta_features = np.column_stack(list(level0_predictions.values()))
    meta_learner = lgb.LGBMClassifier(random_state=42)
    meta_learner.fit(meta_features, y_train)
    
    return meta_learner, level0_predictions

def perform_blending(predictions, X_val, y_val):
    """Blending（単純な平均）を実行"""
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    return ensemble_pred

def perform_dynamic_weighting(predictions, X_val, y_val):
    """動的重み付けを実行"""
    weights = optimize_ensemble_weights_dynamic(predictions, y_val)
    ensemble_pred = np.zeros(len(y_val))
    for i, (model_name, pred) in enumerate(predictions.items()):
        ensemble_pred += weights[i] * pred
    return ensemble_pred

def perform_meta_learning(models, X_train, y_train, X_val):
    """メタ学習を実行"""
    # レベル0の予測を取得
    level0_predictions = {}
    for model_name, model in models.items():
        if hasattr(model, 'predict_proba'):
            level0_predictions[model_name] = model.predict_proba(X_val)[:, 1]
        else:
            level0_predictions[model_name] = model.predict(X_val).flatten()
    
    # メタ特徴量を作成
    meta_features = np.column_stack(list(level0_predictions.values()))
    
    # メタ学習器（XGBoost）
    meta_learner = xgb.XGBClassifier(random_state=42)
    meta_learner.fit(meta_features, y_train)
    
    return meta_learner, level0_predictions

def evaluate_model_performance(y_true, y_pred_proba, threshold=0.5):
    """モデルの性能を評価"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 基本指標
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }
    
    return metrics

def plot_model_performance(y_true, y_pred_proba, fold_num=None):
    """モデルの性能をプロット"""
    # ROC曲線
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # PR曲線
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC曲線
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # PR曲線
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存
    fold_suffix = f"_fold_{fold_num}" if fold_num else ""
    plt.savefig(f'results/performance_curves{fold_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(cv_results, feature_importance=None):
    """詳細な結果を保存"""
    # 結果ディレクトリを作成
    os.makedirs('results', exist_ok=True)
    
    # numpy配列をリストに変換する関数
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # 基本結果を保存
    with open('results/detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(cv_results), f, ensure_ascii=False, indent=2)
    
    # 特徴量重要度を保存
    if feature_importance is not None:
        # 列名を確認して修正
        if 'importance' not in feature_importance.columns and 'gain' in feature_importance.columns:
            feature_importance = feature_importance.rename(columns={'gain': 'importance'})
        elif 'importance' not in feature_importance.columns and 'split' in feature_importance.columns:
            feature_importance = feature_importance.rename(columns={'split': 'importance'})
        
        # 重要度でソート
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance.to_csv('results/feature_importance.csv', index=False)
        
        # 上位20個の特徴量をプロット
        top_features = feature_importance.head(20)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("詳細結果を保存しました: results/")

def save_modelwise_metrics(modelwise_metrics):
    """各モデルごとの評価指標を保存"""
    os.makedirs('results', exist_ok=True)
    
    # JSON形式で保存
    with open('results/modelwise_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(modelwise_metrics, f, ensure_ascii=False, indent=2)
    
    # CSV形式で保存
    df = pd.DataFrame(modelwise_metrics)
    df.to_csv('results/modelwise_metrics.csv', index=False)
    
    # Markdown形式で保存（手動でテーブル作成）
    with open('results/modelwise_metrics.md', 'w', encoding='utf-8') as f:
        f.write('# 各モデルごとの評価指標\n\n')
        f.write('| Fold | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Specificity |\n')
        f.write('|------|-------|---------|--------|-----------|--------|----|-------------|\n')
        for m in modelwise_metrics:
            f.write(f"| {m['fold']} | {m['model']} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['specificity']:.4f} |\n")
    
    print("モデル別評価指標を保存しました: results/")

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去"""
    correlation_matrix = df[feature_columns].corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    return [col for col in feature_columns if col not in to_drop]

def main():
    """メイン実行関数（気象情報のみ使用）"""
    try:
        # データの読み込み
        df = load_processed_data()
        
        # 日付でソート
        df = df.sort_values('date')
        
        # データ期間制限を削除（全期間を使用）
        print(f"全期間データ使用、データ数: {len(df)}")
        print("⚠️ 気象情報のみを使用した東京全体モデルです（入院人数情報は除外）")
        
        # ターゲット変数の作成（75%タイルで閾値を設定）
        threshold = df['hospitalization_count'].quantile(0.75)  # 75%タイル
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        print(f"ターゲット変数作成完了: 75%タイル閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
        
        # 日付関連の特徴量を作成
        df = create_date_features(df)
        
        # 異常気象の検出
        df = detect_extreme_weather(df)
        
        # 特徴量の選択（使用しない列を除外）
        exclude_cols = ['date', 'target', 'season', 'prefecture_name', 'hospitalization_count', 'people_tokyo', 'people_weather']
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
        
        print(f"使用特徴量数: {len(feature_columns)} (気象情報のみ)")
        
        # 数値型の列のみを抽出
        numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
        
        # 欠損値の処理（前方補完のみを使用）
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # 残りの欠損値を中央値で補完
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 無限大の値を適切な値に置換
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
        for fold, (train_dates, test_dates) in enumerate(splits, 1):
            print(f"\nFold {fold}の処理を開始します...")
            
            # データの準備
            X_train, y_train, X_test, y_test, feature_columns = prepare_data_for_training(df, train_dates, test_dates)
            
            # アンサンブルモデルの訓練
            models, predictions, ensemble_pred, optimal_weights = train_ensemble_models(X_train, y_train, X_test, y_test)
            
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
            
            # アンサンブル予測の評価
            ensemble_metrics = evaluate_model_performance(y_test, ensemble_pred)
            cv_results.append(ensemble_metrics)
            
            # 性能曲線のプロット
            plot_model_performance(y_test, ensemble_pred, fold)
            
            # 特徴量の重要度（LightGBMの結果を使用）
            if fold == 1:  # 最初のfoldの特徴量重要度のみを保存
                importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': models['lgb'].feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False)
                feature_importance = importance
        
        # 詳細な結果の保存
        save_detailed_results(cv_results, feature_importance)
        # 各モデルごとの評価指標を保存
        save_modelwise_metrics(modelwise_metrics)
        
        # 最終的な性能の表示
        print("\n=== 最終的な評価結果（気象情報のみ） ===")
        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1_score', 'specificity']
        
        for metric in metrics:
            values = [result[metric] for result in cv_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n東京全体の入院リスク予測モデル（気象情報のみ）の構築が完了しました。")
        print(f"平均ROC-AUC: {np.mean([r['roc_auc'] for r in cv_results]):.4f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 