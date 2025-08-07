import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import jpholiday
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path='../東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv'):
    """データの読み込みと基本的な前処理"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # コロナ禍以前のデータのみを使用
    df = df[df['date'] <= '2019-12-31']
    df = df[df['date'] >= '2012-04-01']
    
    # 日付でソート
    df = df.sort_values('date')
    
    print(f"データ期間: {df['date'].min()} から {df['date'].max()}")
    print(f"データ件数: {len(df)}")
    
    return df

def create_time_features(df):
    """時間関連の特徴量を作成"""
    # 基本的な時間特徴量
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    
    # 休日フラグ
    df['is_holiday'] = df['date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() >= 5)
    )
    
    # 季節性（sin/cosを使用）
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 週末・祝日の連続日数
    df['consecutive_holidays'] = df['is_holiday'].rolling(window=3, min_periods=1).sum()
    
    return df

def create_weather_interaction_features(df):
    """気象要素間の相互作用特徴量を作成"""
    if 'avg_temp' in df.columns and 'avg_humidity' in df.columns:
        # 前日の気象データを使用
        temp = df['avg_temp'].shift(1)
        humidity = df['avg_humidity'].shift(1)
        
        # 不快指数（前日の気温と湿度から計算）
        df['discomfort_index'] = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3
        
        # 気温変化（前日と前々日の差）
        df['temp_change'] = temp - temp.shift(1)
        
        # 過去7日間の平均気温（前日までのデータを使用）
        df['temp_ma_7d'] = temp.rolling(window=7, min_periods=1).mean()
        
        # 過去7日間の気温標準偏差（前日までのデータを使用）
        df['temp_std_7d'] = temp.rolling(window=7, min_periods=1).std()
        
        # 極端気象の指標（前日の気温が過去30日の95パーセンタイルを超えるか）
        df['extreme_temp'] = (temp > temp.rolling(window=30, min_periods=1).quantile(0.95)).astype(int)
        
        # 気温と湿度の相互作用
        df['temp_humidity_interaction'] = temp * humidity
        
        # 気温変化の急激さ（3日間の変化率）
        df['temp_change_rate'] = (temp - temp.shift(3)) / 3
    
    return df

def create_weather_flags(df):
    """異常気象フラグ作成（暴風・台風・急激な気圧変化など）"""
    print("異常気象フラグ作成中...")
    
    # 台風・暴風系フラグ
    if 'pressure_local' in df.columns and 'avg_wind' in df.columns:
        df['pressure_change_24h'] = df['pressure_local'].diff(24)
        df['wind_change_24h'] = df['avg_wind'].diff(24)
        df['typhoon_approach'] = ((df['pressure_change_24h'] < -10) & (df['avg_wind'] > 5)).astype(int)
        df['typhoon_landfall'] = ((df['pressure_change_24h'] < -20) & (df['avg_wind'] > 8)).astype(int)
        df['bomb_cyclone'] = (df['pressure_change_24h'] < -24).astype(int)
        df['strong_wind'] = (df['avg_wind'] > 5).astype(int)
    
    # 急激な気温変化系フラグ
    if 'avg_temp' in df.columns:
        df['temp_change_1d'] = df['avg_temp'].diff(1)
        df['temperature_shock'] = (abs(df['temp_change_1d']) > 7).astype(int)
        df['cold_shock'] = (df['temp_change_1d'] < -7).astype(int)
        df['heat_shock'] = (df['temp_change_1d'] > 7).astype(int)
    
    # 極端な高温・熱帯化フラグ
    if 'max_temp' in df.columns and 'min_temp' in df.columns:
        df['hot_day'] = (df['max_temp'] >= 30).astype(int)
        df['very_hot_day'] = (df['max_temp'] >= 35).astype(int)
        df['tropical_night'] = (df['min_temp'] >= 25).astype(int)
        df['heatwave'] = (df['hot_day'].rolling(window=7, min_periods=1).sum() >= 3).astype(int)
    
    # 極端な低温・寒波フラグ
    if 'min_temp' in df.columns:
        df['cold_day'] = (df['min_temp'] <= 0).astype(int)
        df['very_cold_day'] = (df['min_temp'] <= -5).astype(int)
        df['cold_wave'] = (df['cold_day'].rolling(window=7, min_periods=1).sum() >= 3).astype(int)
    
    # 急激な降雨・雷雨系フラグ
    if 'precipitation' in df.columns:
        df['heavy_rain'] = (df['precipitation'] > 50).astype(int)
        df['very_heavy_rain'] = (df['precipitation'] > 100).astype(int)
        df['rain_shock'] = (df['precipitation'].diff(1) > 30).astype(int)
    
    return df

def create_lagged_features(df):
    """過去の入院データに基づく特徴量を作成（必ず前日以前のデータのみを使用）"""
    # 前日までの入院数の特徴量
    for lag in [1, 2, 3, 7, 14, 28]:
        # ラグ付きの入院数
        df[f'patients_lag_{lag}'] = df['people'].shift(lag)
    
    # 移動平均（前日までのデータを使用）
    for window in [7, 14, 28]:
        # 前日までの移動平均（当日を含まない）
        df[f'patients_ma_{window}'] = df['people'].shift(1).rolling(
            window=window, min_periods=window
        ).mean()
        
        # 前日までの標準偏差（当日を含まない）
        df[f'patients_std_{window}'] = df['people'].shift(1).rolling(
            window=window, min_periods=window
        ).std()
        
        # 前日までの最大値（当日を含まない）
        df[f'patients_max_{window}'] = df['people'].shift(1).rolling(
            window=window, min_periods=window
        ).max()
        
        # 前日までの最小値（当日を含まない）
        df[f'patients_min_{window}'] = df['people'].shift(1).rolling(
            window=window, min_periods=window
        ).min()
    
    # 曜日ごとの過去の統計量（過去28日間のデータのみを使用、当日を含まない）
    df['dow_mean_28d'] = df.groupby('day_of_week')['people'].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=28).mean()
    )
    
    # 月ごとの過去の統計量（過去90日間のデータのみを使用、当日を含まない）
    df['month_mean_90d'] = df.groupby('month')['people'].transform(
        lambda x: x.shift(1).rolling(window=90, min_periods=90).mean()
    )
    
    # トレンド特徴量（過去7日間の傾き）
    df['trend_7d'] = df['people'].shift(1).rolling(window=7, min_periods=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    return df

def handle_outliers(df, columns, n_std=5):
    """外れ値の処理（平均±n標準偏差でクリッピング）"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - n_std * std, mean + n_std * std)
    return df

def create_target_variable(df, window_size=28):
    """目的変数の作成（過去のデータのみを使用）"""
    # 過去window_size日間の移動平均を計算（当日を含まない）
    historical_mean = df['people'].shift(1).rolling(
        window=window_size, min_periods=1
    ).mean()
    
    # 当日の入院数が過去の移動平均より多いかどうか
    df['target'] = (df['people'] > historical_mean).astype(int)
    
    return df

def handle_missing_values(df, columns):
    """欠損値の処理（前方補完 → 後方補完）"""
    df[columns] = df[columns].fillna(method='ffill').fillna(method='bfill')
    return df

def preprocess_data(df):
    """データの前処理を実行"""
    # 時間関連の特徴量を作成
    df = create_time_features(df)
    
    # 気象要素間の相互作用特徴量を作成
    df = create_weather_interaction_features(df)
    
    # 異常気象フラグを作成
    df = create_weather_flags(df)
    
    # 過去の入院データに基づく特徴量を作成
    df = create_lagged_features(df)
    
    # 目的変数の作成（過去28日間の移動平均を使用）
    df = create_target_variable(df, window_size=28)
    
    # 最初の90日分のデータは特徴量の計算に使用したため除外
    df = df.iloc[90:]
    
    # 特徴量の選択（当日のデータを含まない特徴量のみ）
    feature_columns = [col for col in df.columns if col not in [
        'date', 'people', 'target', 'avg_temp', 'avg_humidity'
    ]]
    
    # 数値型の特徴量のみを抽出
    numeric_features = df[feature_columns].select_dtypes(include=['int64', 'float64']).columns
    
    # 欠損値の処理（前方補完のみを使用）
    df[numeric_features] = df[numeric_features].fillna(method='ffill')
    
    # 残りの欠損値を中央値で補完
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    
    return df, list(numeric_features) 