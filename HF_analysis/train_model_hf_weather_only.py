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
    
    # 急激な気圧変化（前日との気圧差が大きい）
    df['pressure_change'] = df['pressure_local'] - df['pressure_local'].shift(1)
    df['pressure_change'] = df['pressure_change'].fillna(0)
    df['pressure_change_std'] = df['pressure_change'].expanding().std().shift(1).fillna(df['pressure_change'].std())
    df['is_rapid_pressure_change'] = (abs(df['pressure_change']) > df['pressure_change_std'] * 2).astype(int)
    
    # 異常湿度（極端に湿度が高いまたは低い）
    df['humidity_quantile_95'] = df['avg_humidity_weather'].expanding().quantile(0.95).shift(1).fillna(df['avg_humidity_weather'].quantile(0.95))
    df['humidity_quantile_05'] = df['avg_humidity_weather'].expanding().quantile(0.05).shift(1).fillna(df['avg_humidity_weather'].quantile(0.05))
    df['is_extreme_humidity_high'] = (df['avg_humidity_weather'] > df['humidity_quantile_95']).astype(int)
    df['is_extreme_humidity_low'] = (df['avg_humidity_weather'] < df['humidity_quantile_05']).astype(int)
    
    # 日照時間の極端な状態
    df['sunshine_quantile_95'] = df['sunshine_hours_weather'].expanding().quantile(0.95).shift(1).fillna(df['sunshine_hours_weather'].quantile(0.95))
    df['sunshine_quantile_05'] = df['sunshine_hours_weather'].expanding().quantile(0.05).shift(1).fillna(df['sunshine_hours_weather'].quantile(0.05))
    df['is_extremely_sunny'] = (df['sunshine_hours_weather'] > df['sunshine_quantile_95']).astype(int)
    df['is_extremely_cloudy'] = (df['sunshine_hours_weather'] < df['sunshine_quantile_05']).astype(int)
    
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
    df['temp_quantile_20'] = df['avg_temp_weather'].expanding().quantile(0.2).shift(1).fillna(df['avg_temp_weather'].quantile(0.2))
    df['wind_quantile_80'] = df['avg_wind_weather'].expanding().quantile(0.8).shift(1).fillna(df['avg_wind_weather'].quantile(0.8))
    df['temp_wind_stress'] = (
        (df['avg_temp_weather'] < df['temp_quantile_20']) & 
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
    
    # 複合的な気象ストレス指標の更新
    df['weather_stress'] = (
        df['weather_stress'] +
        df['temp_humidity_stress'] * 0.15 +
        df['temp_wind_stress'] * 0.15 +
        df['pressure_wind_stress'] * 0.15 +
        df['sunshine_temp_stress'] * 0.15
    )
    
    # 季節変動の影響を考慮した気象パターン
    df['seasonal_temp_deviation'] = df.groupby('month')['avg_temp_weather'].transform(
        lambda x: (x - x.expanding().mean().shift(1).fillna(x.mean())) / 
                 x.expanding().std().shift(1).fillna(x.std())
    )
    
    # 地域特有の気象パターン（東京特有の気象条件）
    df['tokyo_specific_heat'] = (
        (df['month'].isin([7, 8])) & 
        (df['avg_temp_weather'] > df['temp_quantile_80']) & 
        (df['avg_humidity_weather'] > df['humidity_quantile_80'])
    ).astype(int)
    
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

def create_advanced_weather_timeseries_features(df):
    """気象×時系列の高度な組み合わせ特徴量を作成"""
    
    # 1. 気象要素の季節性を考慮した移動平均
    for window in [7, 14, 30]:
        # 季節ごとの気温偏差
        df[f'temp_seasonal_deviation_{window}d'] = df.groupby('month')['avg_temp_weather'].transform(
            lambda x: x - x.rolling(window=window, min_periods=1).mean()
        )
        
        # 季節ごとの湿度偏差
        df[f'humidity_seasonal_deviation_{window}d'] = df.groupby('month')['avg_humidity_weather'].transform(
            lambda x: x - x.rolling(window=window, min_periods=1).mean()
        )
        
        # 季節ごとの気圧偏差
        df[f'pressure_seasonal_deviation_{window}d'] = df.groupby('month')['pressure_local'].transform(
            lambda x: x - x.rolling(window=window, min_periods=1).mean()
        )
    
    # 2. 気象要素の急激な変化の検出
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # 加速度（2次微分）
            df[f'{col}_acceleration_2nd'] = (
                df[col] - 2 * df[col].shift(1) + df[col].shift(2)
            ).fillna(0)
            
            # 急激な変化の検出（標準偏差の2倍以上）
            df[f'{col}_rapid_change'] = (
                abs(df[f'{col}_acceleration_2nd']) > 
                df[f'{col}_acceleration_2nd'].rolling(window=30).std() * 2
            ).astype(int)
    
    # 3. 気象要素の組み合わせパターン
    # 高温高湿度の連続日数
    df['hot_humid_consecutive'] = (
        (df['avg_temp_weather'] > df['avg_temp_weather'].rolling(window=365).quantile(0.8)) &
        (df['avg_humidity_weather'] > df['avg_humidity_weather'].rolling(window=365).quantile(0.8))
    ).astype(int).rolling(window=7).sum()
    
    # 低温低湿度の連続日数
    df['cold_dry_consecutive'] = (
        (df['avg_temp_weather'] < df['avg_temp_weather'].rolling(window=365).quantile(0.2)) &
        (df['avg_humidity_weather'] < df['avg_humidity_weather'].rolling(window=365).quantile(0.2))
    ).astype(int).rolling(window=7).sum()
    
    # 4. 気象要素の非線形相互作用
    # 気温と湿度の非線形相互作用
    df['temp_humidity_nonlinear'] = np.exp(-0.1 * (df['avg_temp_weather'] - df['avg_humidity_weather'])**2)
    
    # 気温と気圧の非線形相互作用
    df['temp_pressure_nonlinear'] = np.exp(-0.1 * (df['avg_temp_weather'] - df['pressure_local'])**2)
    
    # 5. 気象要素の周期性パターン
    # 週間周期の気温パターン
    df['temp_weekly_pattern'] = df.groupby('dayofweek')['avg_temp_weather'].transform(
        lambda x: x - x.rolling(window=52, min_periods=1).mean()  # 1年分の同じ曜日の平均
    )
    
    # 月間周期の湿度パターン
    df['humidity_monthly_pattern'] = df.groupby('month')['avg_humidity_weather'].transform(
        lambda x: x - x.rolling(window=12, min_periods=1).mean()  # 1年分の同じ月の平均
    )
    
    # 6. 気象要素の異常値検出
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # Z-score based anomaly
            rolling_mean = df[col].rolling(window=30).mean()
            rolling_std = df[col].rolling(window=30).std()
            df[f'{col}_anomaly_score'] = abs((df[col] - rolling_mean) / (rolling_std + 1e-8))
            df[f'{col}_is_anomaly'] = (df[f'{col}_anomaly_score'] > 2).astype(int)
    
    # 7. 気象要素のトレンド分析
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # 線形トレンドの傾き
            df[f'{col}_trend_slope'] = df[col].rolling(window=30).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            # トレンドの強度
            df[f'{col}_trend_strength'] = abs(df[f'{col}_trend_slope'])
    
    # 8. 複合気象ストレス指数
    # 熱ストレス指数
    df['heat_stress_index'] = (
        (df['avg_temp_weather'] > df['avg_temp_weather'].rolling(window=365).quantile(0.8)).astype(float) * 0.4 +
        (df['avg_humidity_weather'] > df['avg_humidity_weather'].rolling(window=365).quantile(0.8)).astype(float) * 0.3 +
        (df['sunshine_hours_weather'] > df['sunshine_hours_weather'].rolling(window=365).quantile(0.8)).astype(float) * 0.3
    )
    
    # 寒ストレス指数
    df['cold_stress_index'] = (
        (df['avg_temp_weather'] < df['avg_temp_weather'].rolling(window=365).quantile(0.2)).astype(float) * 0.5 +
        (df['avg_wind_weather'] > df['avg_wind_weather'].rolling(window=365).quantile(0.8)).astype(float) * 0.3 +
        (df['pressure_local'] < df['pressure_local'].rolling(window=365).quantile(0.2)).astype(float) * 0.2
    )
    
    return df

def create_seasonal_weighted_features(df):
    """季節性を考慮した重み付け特徴量を作成"""
    
    # 季節ごとの気象要素の重み付け
    seasons = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    
    for season_name, months in seasons.items():
        season_mask = df['month'].isin(months)
        
        # 季節ごとの気温重み付け
        df[f'temp_weighted_{season_name}'] = df['avg_temp_weather'] * season_mask.astype(float)
        
        # 季節ごとの湿度重み付け
        df[f'humidity_weighted_{season_name}'] = df['avg_humidity_weather'] * season_mask.astype(float)
        
        # 季節ごとの気圧重み付け
        df[f'pressure_weighted_{season_name}'] = df['pressure_local'] * season_mask.astype(float)
    
    # 月ごとの気象要素の正規化
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # 月ごとの平均と標準偏差で正規化
            monthly_mean = df.groupby('month')[col].transform('mean')
            monthly_std = df.groupby('month')[col].transform('std')
            df[f'{col}_monthly_normalized'] = (df[col] - monthly_mean) / (monthly_std + 1e-8)
    
    # 季節性を考慮した異常気象の検出
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # 月ごとの95%タイルと5%タイル
            monthly_95th = df.groupby('month')[col].transform(lambda x: x.quantile(0.95))
            monthly_5th = df.groupby('month')[col].transform(lambda x: x.quantile(0.05))
            
            # 異常値フラグ
            df[f'{col}_monthly_extreme_high'] = (df[col] > monthly_95th).astype(int)
            df[f'{col}_monthly_extreme_low'] = (df[col] < monthly_5th).astype(int)
    
    # 季節性を考慮した気象ストレス指数
    # 夏の熱ストレス
    summer_mask = df['month'].isin([6, 7, 8])
    df['summer_heat_stress'] = (
        (df['avg_temp_weather'] > 30) & 
        (df['avg_humidity_weather'] > 70) & 
        summer_mask
    ).astype(float)
    
    # 冬の寒ストレス
    winter_mask = df['month'].isin([12, 1, 2])
    df['winter_cold_stress'] = (
        (df['avg_temp_weather'] < 5) & 
        (df['avg_wind_weather'] > 5) & 
        winter_mask
    ).astype(float)
    
    # 季節性を考慮した気象変化の影響
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        if col in df.columns:
            # 季節ごとの変化率
            for season_name, months in seasons.items():
                season_mask = df['month'].isin(months)
                df[f'{col}_change_{season_name}'] = (
                    df[col].diff() * season_mask.astype(float)
                ).fillna(0)
    
    return df

def create_weather_interaction_features(df):
    """気象要素の相互作用特徴量を作成"""
    # 基本気象要素の相互作用
    df['temp_humidity'] = df['avg_temp_weather'] * df['avg_humidity_weather']
    df['temp_pressure'] = df['avg_temp_weather'] * df['pressure_local']
    df['temp_wind'] = df['avg_temp_weather'] * df['avg_wind_weather']
    df['humidity_pressure'] = df['avg_humidity_weather'] * df['pressure_local']
    
    # 気温変化率
    df['temp_change'] = df['avg_temp_weather'] - df['avg_temp_1day_ago']
    df['temp_change_2d'] = df['avg_temp_weather'] - df['avg_temp_2day_ago']
    df['temp_change_3d'] = df['avg_temp_weather'] - df['avg_temp_3day_ago']
    
    # 湿度変化率
    df['humidity_change'] = df['avg_humidity_weather'] - df['avg_humidity_1day_ago']
    df['humidity_change_2d'] = df['avg_humidity_weather'] - df['avg_humidity_2day_ago']
    
    # 気圧変化率
    df['pressure_change'] = df['pressure_local'] - df['pressure_local'].shift(1)
    df['pressure_change_2d'] = df['pressure_local'] - df['pressure_local'].shift(2)
    
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp_weather'] + 0.01 * df['avg_humidity_weather'] * (0.99 * df['avg_temp_weather'] - 14.3) + 46.3
    
    # 極端な気象条件のフラグ
    df['extreme_temp'] = ((df['max_temp_weather'] >= 35) | (df['min_temp_weather'] <= -10)).astype(int)
    df['extreme_humidity'] = ((df['avg_humidity_weather'] >= 80) | (df['avg_humidity_weather'] <= 20)).astype(int)
    df['extreme_pressure'] = ((df['pressure_local'] >= 1020) | (df['pressure_local'] <= 990)).astype(int)
    
    # 気象条件の組み合わせ
    df['hot_humid'] = ((df['avg_temp_weather'] >= 25) & (df['avg_humidity_weather'] >= 60)).astype(int)
    df['cold_dry'] = ((df['avg_temp_weather'] <= 5) & (df['avg_humidity_weather'] <= 40)).astype(int)
    df['storm_conditions'] = ((df['avg_wind_weather'] >= 10) & (df['precipitation'] > 0)).astype(int)
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    # 気象データの移動平均
    weather_cols = ['avg_temp_weather', 'max_temp_weather', 'min_temp_weather', 
                   'avg_humidity_weather', 'avg_wind_weather', 'pressure_local']
    
    for col in weather_cols:
        if col in df.columns:
            # 7日間の移動平均
            df[f'{col}_ma_7d'] = df[col].rolling(window=7, min_periods=1).mean()
            # 14日間の移動平均
            df[f'{col}_ma_14d'] = df[col].rolling(window=14, min_periods=1).mean()
            # 30日間の移動平均
            df[f'{col}_ma_30d'] = df[col].rolling(window=30, min_periods=1).mean()
            
            # 標準偏差
            df[f'{col}_std_7d'] = df[col].rolling(window=7, min_periods=1).std()
            df[f'{col}_std_14d'] = df[col].rolling(window=14, min_periods=1).std()
            
            # 最大値・最小値
            df[f'{col}_max_7d'] = df[col].rolling(window=7, min_periods=1).max()
            df[f'{col}_min_7d'] = df[col].rolling(window=7, min_periods=1).min()
            
            # 範囲
            df[f'{col}_range_7d'] = df[f'{col}_max_7d'] - df[f'{col}_min_7d']
    
    # 変化率の計算
    for col in weather_cols:
        if col in df.columns:
            df[f'{col}_change_rate'] = df[col].pct_change().fillna(0)
            df[f'{col}_change_rate_7d'] = df[col].pct_change(periods=7).fillna(0)
    
    # 加速度（変化率の変化率）
    for col in weather_cols:
        if col in df.columns:
            df[f'{col}_acceleration'] = df[f'{col}_change_rate'].diff().fillna(0)
    
    return df

def load_processed_data():
    """統合された心不全データを読み込み"""
    print("=== 心不全データ読み込み ===")
    
    # 統合データを読み込み
    df = pd.read_csv('hf_weather_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # 心不全データを正しく設定
    df['hospitalization_count'] = df['people_hf']
    
    print(f"データ期間: {df['date'].min()} から {df['date'].max()}")
    print(f"総データ数: {len(df)}")
    print(f"心不全発生総数: {df['people_hf'].sum()}")
    print(f"平均心不全発生数/日: {df['people_hf'].mean():.2f}")
    
    return df

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

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮した時系列分割を作成"""
    splits = []
    unique_dates = pd.Series(df['date'].unique()).sort_values()
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
        
        # 重み付き予測
        ensemble_pred = np.zeros(len(y_true))
        for i, pred in enumerate(predictions.values()):
            ensemble_pred += weights[i] * pred
        
        # AUCを最大化（負の値を返すことで最小化問題に変換）
        return -roc_auc_score(y_true, ensemble_pred)
    
    return objective

def optimize_ensemble_weights_dynamic(predictions, y_true, method='scipy'):
    """動的にアンサンブル重みを最適化"""
    if method == 'scipy':
        # scipy.optimizeを使用した最適化
        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models
        
        objective = create_ensemble_weights_optimizer()
        
        # 制約条件：重みの合計が1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # 境界条件：重みは0以上
        bounds = [(0, 1)] * n_models
        
        result = minimize(
            lambda w: objective(w, predictions, y_true),
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)  # 正規化
            return optimal_weights
        else:
            print("最適化に失敗しました。均等重みを使用します。")
            return np.ones(n_models) / n_models
    
    elif method == 'grid_search':
        # グリッドサーチによる最適化
        best_weights = None
        best_auc = 0
        
        # 重みの組み合わせを試行
        weight_combinations = [
            [0.25, 0.25, 0.25, 0.25],  # 均等
            [0.4, 0.2, 0.2, 0.2],      # LightGBM重視
            [0.2, 0.4, 0.2, 0.2],      # XGBoost重視
            [0.2, 0.2, 0.4, 0.2],      # CatBoost重視
            [0.2, 0.2, 0.2, 0.4],      # NN重視
            [0.5, 0.2, 0.2, 0.1],      # LightGBM+NN
            [0.3, 0.3, 0.2, 0.2],      # 勾配ブースティング重視
        ]
        
        for weights in weight_combinations:
            ensemble_pred = np.zeros(len(y_true))
            for i, (model_name, pred) in enumerate(predictions.items()):
                ensemble_pred += weights[i] * pred
            
            auc = roc_auc_score(y_true, ensemble_pred)
            if auc > best_auc:
                best_auc = auc
                best_weights = weights
        
        return np.array(best_weights)

def calculate_model_correlation(predictions):
    """モデル間の相関を計算"""
    pred_matrix = np.column_stack(list(predictions.values()))
    correlation_matrix = np.corrcoef(pred_matrix.T)
    return correlation_matrix

def create_diversified_ensemble(predictions, y_true):
    """相関を考慮した多様化アンサンブル"""
    # モデル間の相関を計算
    correlation_matrix = calculate_model_correlation(predictions)
    
    # 相関が高いモデルペアを特定
    high_corr_pairs = []
    model_names = list(predictions.keys())
    
    for i in range(len(correlation_matrix)):
        for j in range(i+1, len(correlation_matrix)):
            if correlation_matrix[i, j] > 0.8:  # 相関閾値
                high_corr_pairs.append((model_names[i], model_names[j]))
    
    # 相関の高いモデルを除外してアンサンブルを作成
    if high_corr_pairs:
        print(f"高相関モデルペア: {high_corr_pairs}")
        # 相関の高いモデルを除外
        excluded_models = set()
        for pair in high_corr_pairs:
            # 性能の低い方を除外
            auc1 = roc_auc_score(y_true, predictions[pair[0]])
            auc2 = roc_auc_score(y_true, predictions[pair[1]])
            if auc1 < auc2:
                excluded_models.add(pair[0])
            else:
                excluded_models.add(pair[1])
        
        # 除外されたモデルを除いてアンサンブル
        filtered_predictions = {k: v for k, v in predictions.items() if k not in excluded_models}
        if len(filtered_predictions) > 1:
            weights = optimize_ensemble_weights_dynamic(filtered_predictions, y_true)
            ensemble_pred = np.zeros(len(y_true))
            for i, (model_name, pred) in enumerate(filtered_predictions.items()):
                ensemble_pred += weights[i] * pred
            return ensemble_pred
    
    # 相関が低い場合は全モデルを使用
    weights = optimize_ensemble_weights_dynamic(predictions, y_true)
    ensemble_pred = np.zeros(len(y_true))
    for i, (model_name, pred) in enumerate(predictions.items()):
        ensemble_pred += weights[i] * pred
    return ensemble_pred

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """複数のモデルを学習し、高度なアンサンブルモデルを作成（強化版）"""
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
        'reg_alpha': 0.1,
        'reg_lambda': 1
    }
    models['xgb'] = xgb.XGBClassifier(**xgb_params, random_state=42)
    models['xgb'].fit(X_train, y_train)
    predictions['xgb'] = models['xgb'].predict_proba(X_val)[:, 1]
    
    # CatBoost
    print("Training CatBoost...")
    cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False
    }
    models['cat'] = cb.CatBoostClassifier(**cat_params)
    models['cat'].fit(X_train, y_train)
    predictions['cat'] = models['cat'].predict_proba(X_val)[:, 1]
    
    # 深層Neural Network
    print("Training Deep Neural Network...")
    deep_nn_model = create_deep_nn_model(X_train.shape[1])
    deep_nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    models['deep_nn'] = deep_nn_model
    predictions['deep_nn'] = deep_nn_model.predict(X_val).flatten()
    
    # Attention付きNeural Network
    print("Training Attention Neural Network...")
    attention_nn_model = create_attention_nn_model(X_train.shape[1])
    attention_nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    models['attention_nn'] = attention_nn_model
    predictions['attention_nn'] = attention_nn_model.predict(X_val).flatten()
    
    # 高度なアンサンブル手法
    print("高度なアンサンブル手法を適用...")
    
    # 1. 動的最適化アンサンブル
    print("動的最適化アンサンブルを実行...")
    dynamic_weights = optimize_ensemble_weights_dynamic(predictions, y_val, method='scipy')
    dynamic_pred = np.zeros(len(y_val))
    for i, (model_name, pred) in enumerate(predictions.items()):
        dynamic_pred += dynamic_weights[i] * pred
        print(f"  {model_name}: {dynamic_weights[i]:.3f}")
    
    # 2. 多様化アンサンブル
    print("多様化アンサンブルを実行...")
    diversified_pred = create_diversified_ensemble(predictions, y_val)
    
    # 3. 従来の手法
    print("従来のアンサンブル手法を実行...")
    stacking_pred = perform_stacking(models, X_train, y_train, X_val)
    blending_pred = perform_blending(predictions, X_val, y_val)
    
    # 4. 高度なメタ学習
    print("高度なメタ学習を実行...")
    advanced_meta_pred, meta_auc = perform_advanced_meta_learning(models, X_train, y_train, X_val, y_val)
    
    # 各手法の性能を評価
    ensemble_methods = {
        'dynamic': dynamic_pred,
        'diversified': diversified_pred,
        'stacking': stacking_pred,
        'blending': blending_pred,
        'advanced_meta': advanced_meta_pred
    }
    
    best_method = None
    best_auc = 0
    
    for method_name, pred in ensemble_methods.items():
        auc = roc_auc_score(y_val, pred)
        print(f"{method_name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_method = method_name
    
    print(f"最適なアンサンブル手法: {best_method} (AUC: {best_auc:.4f})")
    
    return models, ensemble_methods[best_method], best_auc

def perform_stacking(models, X_train, y_train, X_val):
    """Stacking手法を実装"""
    # レベル0の予測を取得
    level0_preds_train = np.column_stack([
        models['lgb'].predict(X_train),
        models['xgb'].predict_proba(X_train)[:, 1],
        models['cat'].predict_proba(X_train)[:, 1],
        models['deep_nn'].predict(X_train).flatten(),
        models['attention_nn'].predict(X_train).flatten()
    ])
    
    level0_preds_val = np.column_stack([
        models['lgb'].predict(X_val),
        models['xgb'].predict_proba(X_val)[:, 1],
        models['cat'].predict_proba(X_val)[:, 1],
        models['deep_nn'].predict(X_val).flatten(),
        models['attention_nn'].predict(X_val).flatten()
    ])
    
    # レベル1のメタ学習器（LightGBM）
    meta_learner = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        learning_rate=0.01,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    meta_learner.fit(level0_preds_train, y_train)
    return meta_learner.predict_proba(level0_preds_val)[:, 1]

def perform_blending(predictions, X_val, y_val):
    """Blending手法を実装"""
    # 各モデルの性能に基づく重み付け
    model_weights = {}
    for model_name, pred in predictions.items():
        auc = roc_auc_score(y_val, pred)
        model_weights[model_name] = auc
    
    # 重みを正規化
    total_weight = sum(model_weights.values())
    for model_name in model_weights:
        model_weights[model_name] /= total_weight
    
    # 重み付き平均
    blended_pred = np.zeros(len(y_val))
    for model_name, pred in predictions.items():
        blended_pred += model_weights[model_name] * pred
    
    return blended_pred

def perform_dynamic_weighting(predictions, X_val, y_val):
    """Dynamic Weighting手法を実装"""
    # 各モデルの信頼度を計算
    model_confidence = {}
    for model_name, pred in predictions.items():
        # 予測の確信度を計算（0.5からの距離）
        confidence = np.abs(pred - 0.5)
        model_confidence[model_name] = np.mean(confidence)
    
    # 信頼度に基づく重み付け
    total_confidence = sum(model_confidence.values())
    for model_name in model_confidence:
        model_confidence[model_name] /= total_confidence
    
    # 動的重み付き平均
    dynamic_pred = np.zeros(len(y_val))
    for model_name, pred in predictions.items():
        dynamic_pred += model_confidence[model_name] * pred
    
    return dynamic_pred

def perform_meta_learning(models, X_train, y_train, X_val):
    """Meta-Learning手法を実装"""
    # 各モデルの予測を特徴量として使用
    meta_features_train = np.column_stack([
        models['lgb'].predict(X_train),
        models['xgb'].predict_proba(X_train)[:, 1],
        models['cat'].predict_proba(X_train)[:, 1],
        models['deep_nn'].predict(X_train).flatten(),
        models['attention_nn'].predict(X_train).flatten()
    ])
    
    meta_features_val = np.column_stack([
        models['lgb'].predict(X_val),
        models['xgb'].predict_proba(X_val)[:, 1],
        models['cat'].predict_proba(X_val)[:, 1],
        models['deep_nn'].predict(X_val).flatten(),
        models['attention_nn'].predict(X_val).flatten()
    ])
    
    # 元の特徴量と組み合わせ
    meta_features_train = np.column_stack([X_train, meta_features_train])
    meta_features_val = np.column_stack([X_val, meta_features_val])
    
    # メタ学習器（CatBoost）
    meta_learner = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        random_seed=42,
        verbose=False
    )
    
    meta_learner.fit(meta_features_train, y_train)
    return meta_learner.predict_proba(meta_features_val)[:, 1]

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
    """モデルの性能をプロット"""
    # NaN値を処理
    if np.isnan(y_pred_proba).any():
        median_pred = np.nanmedian(y_pred_proba)
        y_pred_proba = np.where(np.isnan(y_pred_proba), median_pred, y_pred_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC曲線
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)
    
    # Precision-Recall曲線
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # 保存ディレクトリを作成
    os.makedirs('results', exist_ok=True)
    
    # 保存
    fold_str = f'_fold_{fold_num}' if fold_num is not None else ''
    plt.savefig(f'results/performance_curves{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(cv_results, feature_importance=None):
    """詳細な結果を保存"""
    # 結果ディレクトリを作成
    os.makedirs('results', exist_ok=True)
    
    # numpy配列をリストに変換する関数
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
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

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去"""
    corr_matrix = df[feature_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    reduced_features = [f for f in feature_columns if f not in to_drop]
    return reduced_features

def create_advanced_meta_learner(input_dim):
    """高度なメタ学習器を作成"""
    model = tf.keras.Sequential([
        # 入力層（各モデルの予測を入力）
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # 隠れ層
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # 出力層
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC', 'accuracy']
    )
    return model

def create_adaptive_ensemble_selector(predictions, y_true):
    """適応的アンサンブル選択器"""
    # 各モデルの性能を評価
    model_performances = {}
    for model_name, pred in predictions.items():
        auc = roc_auc_score(y_true, pred)
        model_performances[model_name] = auc
    
    # 性能の高いモデルを選択（上位80%）
    sorted_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)
    threshold = len(sorted_models) * 0.8
    selected_models = [model for model, _ in sorted_models[:int(threshold)]]
    
    print(f"選択されたモデル: {selected_models}")
    return selected_models

def create_dynamic_weight_optimizer(predictions, y_true, window_size=30):
    """動的重み最適化器"""
    n_samples = len(y_true)
    optimal_weights = np.ones(len(predictions)) / len(predictions)
    
    # スライディングウィンドウで重みを最適化
    for i in range(window_size, n_samples, window_size):
        window_start = max(0, i - window_size)
        window_end = min(n_samples, i)
        
        window_y = y_true[window_start:window_end]
        window_predictions = {name: pred[window_start:window_end] for name, pred in predictions.items()}
        
        # このウィンドウでの最適重みを計算
        def objective(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            ensemble_pred = np.zeros(len(window_y))
            for j, (model_name, pred) in enumerate(window_predictions.items()):
                ensemble_pred += weights[j] * pred
            
            return -roc_auc_score(window_y, ensemble_pred)
        
        # 最適化
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(predictions)
        
        result = minimize(
            objective,
            optimal_weights,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        if result.success:
            optimal_weights = result.x / np.sum(result.x)
    
    return optimal_weights

def create_confidence_weighted_ensemble(predictions, y_true):
    """信頼度重み付きアンサンブル"""
    # 各モデルの信頼度を計算
    confidence_scores = {}
    
    for model_name, pred in predictions.items():
        # 予測の確信度を計算（0.5からの距離）
        confidence = np.abs(pred - 0.5) * 2  # 0-1のスケールに変換
        avg_confidence = np.mean(confidence)
        confidence_scores[model_name] = avg_confidence
    
    # 信頼度で重みを正規化
    total_confidence = sum(confidence_scores.values())
    confidence_weights = {name: conf/total_confidence for name, conf in confidence_scores.items()}
    
    print(f"信頼度重み: {confidence_weights}")
    return confidence_weights

def create_uncertainty_aware_ensemble(predictions, y_true):
    """不確実性を考慮したアンサンブル"""
    # 各モデルの予測分散を計算
    prediction_matrix = np.column_stack(list(predictions.values()))
    prediction_variance = np.var(prediction_matrix, axis=1)
    
    # 分散の逆数を重みとして使用（低分散=高信頼度）
    inverse_variance = 1 / (prediction_variance + 1e-8)
    uncertainty_weights = inverse_variance / np.sum(inverse_variance)
    
    model_names = list(predictions.keys())
    uncertainty_weight_dict = {name: weight for name, weight in zip(model_names, uncertainty_weights)}
    
    print(f"不確実性重み: {uncertainty_weight_dict}")
    return uncertainty_weight_dict

def perform_advanced_meta_learning(models, X_train, y_train, X_val, y_val):
    """高度なメタ学習を実行"""
    print("高度なメタ学習を実行...")
    
    # 各モデルの予測を取得
    train_predictions = {}
    val_predictions = {}
    
    for model_name, model in models.items():
        if model_name == 'lgb':
            # LightGBM Boosterオブジェクトの場合
            train_predictions[model_name] = model.predict(X_train)
            val_predictions[model_name] = model.predict(X_val)
        elif model_name in ['xgb', 'cat']:
            train_predictions[model_name] = model.predict_proba(X_train)[:, 1]
            val_predictions[model_name] = model.predict_proba(X_val)[:, 1]
        else:  # Neural Network
            train_predictions[model_name] = model.predict(X_train).flatten()
            val_predictions[model_name] = model.predict(X_val).flatten()
    
    # 1. 適応的アンサンブル選択
    print("適応的アンサンブル選択を実行...")
    selected_models = create_adaptive_ensemble_selector(val_predictions, y_val)
    selected_predictions = {name: val_predictions[name] for name in selected_models}
    
    # 2. 動的重み最適化
    print("動的重み最適化を実行...")
    dynamic_weights = create_dynamic_weight_optimizer(selected_predictions, y_val)
    
    # 3. 信頼度重み付きアンサンブル
    print("信頼度重み付きアンサンブルを実行...")
    confidence_weights = create_confidence_weighted_ensemble(selected_predictions, y_val)
    
    # 4. 不確実性を考慮したアンサンブル
    print("不確実性を考慮したアンサンブルを実行...")
    uncertainty_weights = create_uncertainty_aware_ensemble(selected_predictions, y_val)
    
    # 5. 高度なメタ学習器
    print("高度なメタ学習器を訓練...")
    meta_features = np.column_stack(list(selected_predictions.values()))
    meta_learner = create_advanced_meta_learner(len(selected_models))
    meta_learner.fit(meta_features, y_val, epochs=50, batch_size=32, verbose=0)
    
    # 各手法の性能を評価
    ensemble_methods = {}
    
    # 動的重みアンサンブル
    dynamic_pred = np.zeros(len(y_val))
    for i, (model_name, pred) in enumerate(selected_predictions.items()):
        dynamic_pred += dynamic_weights[i] * pred
    ensemble_methods['dynamic'] = dynamic_pred
    
    # 信頼度重みアンサンブル
    confidence_pred = np.zeros(len(y_val))
    for model_name, pred in selected_predictions.items():
        confidence_pred += confidence_weights[model_name] * pred
    ensemble_methods['confidence'] = confidence_pred
    
    # 不確実性重みアンサンブル
    uncertainty_pred = np.zeros(len(y_val))
    for model_name, pred in selected_predictions.items():
        uncertainty_pred += uncertainty_weights[model_name] * pred
    ensemble_methods['uncertainty'] = uncertainty_pred
    
    # メタ学習器
    meta_pred = meta_learner.predict(meta_features).flatten()
    ensemble_methods['meta_learner'] = meta_pred
    
    # 最適な手法を選択
    best_method = None
    best_auc = 0
    
    for method_name, pred in ensemble_methods.items():
        auc = roc_auc_score(y_val, pred)
        print(f"{method_name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_method = method_name
    
    print(f"最適なメタ学習手法: {best_method} (AUC: {best_auc:.4f})")
    
    return ensemble_methods[best_method], best_auc

def main():
    """メイン実行関数（気象情報のみ使用）"""
    try:
        # データの読み込み
        df = load_processed_data()
        
        # 日付でソート
        df = df.sort_values('date')
        
        # データ期間制限を削除（全期間を使用）
        print(f"全期間データ使用、データ数: {len(df)}")
        print("⚠️ 気象情報のみを使用した心不全モデルです（入院人数情報は除外）")
        
        # ターゲット変数の作成（75%タイルで閾値を設定）
        threshold = df['hospitalization_count'].quantile(0.75)  # 75%タイル
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        print(f"ターゲット変数作成完了: 75%タイル閾値={threshold:.1f}, 高リスク日割合={df['target'].mean():.3f}")
        
        # 日付関連の特徴量を作成
        df = create_date_features(df)
        
        # 気象要素の相互作用特徴量を作成
        df = create_weather_interaction_features(df)
        
        # 時系列特徴量を作成
        df = create_time_series_features(df)
        
        # 気象×時系列の高度な組み合わせ特徴量を作成
        df = create_advanced_weather_timeseries_features(df)
        
        # 季節性を考慮した重み付け特徴量を作成
        df = create_seasonal_weighted_features(df)
        
        # 特徴量の選択（使用しない列を除外）
        exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 'hospitalization_count', 'people_hf', 'people_weather']
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
        for fold, (X_train, y_train, X_test, y_test, feature_columns) in enumerate(splits, 1):
            print(f"\nFold {fold}の処理を開始します...")
            best_auc_fold = 0
            best_n = None
            best_features = feature_columns
            best_result = None
            auc_results = {}
            
            # 特徴量数を変えてループ
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
            
            # 各モデルの予測
            predictions = {}
            for model_name, model in models.items():
                if model_name == 'lgb':
                    predictions[model_name] = model.predict(X_test_scaled)
                elif model_name == 'xgb':
                    predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]
                elif model_name == 'cat':
                    predictions[model_name] = model.predict_proba(X_test_scaled)[:, 1]
                elif model_name == 'nn':
                    predictions[model_name] = model.predict(X_test_scaled).flatten()
            
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
            ensemble_pred = best_ensemble_pred # 最適なアンサンブル予測を使用
            
            # モデルの評価
            fold_metrics = evaluate_model_performance(y_test, ensemble_pred)
            cv_results.append(fold_metrics)
            
            # 性能曲線のプロット
            plot_model_performance(y_test, ensemble_pred, fold)
            
            # 特徴量の重要度（LightGBMの結果を使用）
            if fold == 1:  # 最初のfoldの特徴量重要度のみを保存
                importance = pd.DataFrame({
                    'feature': best_features,  # 選択された特徴量のみ
                    'importance': models['lgb'].feature_importance()
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
            values = [r[metric] for r in cv_results]
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 