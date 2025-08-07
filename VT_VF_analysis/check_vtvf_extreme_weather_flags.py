import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_vtvf_data():
    """VT/VFデータを読み込み"""
    df = pd.read_csv('vtvf_weather_merged.csv')
    df['hospitalization_date'] = pd.to_datetime(df['hospitalization_date_vtvf'])
    df['hospitalization_count'] = df['people_vtvf']
    return df

def create_date_features(df):
    """日付関連の特徴量を作成"""
    from datetime import datetime, timedelta
    import jpholiday
    
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end.astype(int)
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def create_weather_interaction_features(df):
    """気象相互作用特徴量を作成"""
    # 異常気象フラグ
    df['is_tropical_night'] = (df['min_temp_vtvf'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp_vtvf'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp_vtvf'] >= 30).astype(int)
    df['is_summer_day'] = (df['max_temp_vtvf'] >= 25).astype(int)
    df['is_winter_day'] = (df['min_temp_vtvf'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp_vtvf'] < 0).astype(int)
    
    # 気象相互作用
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

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成"""
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        df[f'{col}_change_rate'] = df[col].pct_change()
        
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

def prepare_features(df):
    """特徴量を準備"""
    df = create_date_features(df)
    df = create_weather_interaction_features(df)
    df = create_time_series_features(df)
    df = create_advanced_weather_timeseries_features(df)
    df = create_seasonal_weighted_features(df)
    
    # ターゲット変数
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    return df

def select_features(df):
    """特徴量を選択"""
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name_vtvf', 'date', 
                   'hospitalization_count', 'people_vtvf']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    return feature_columns

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去"""
    if len(feature_columns) <= 1:
        return feature_columns
    
    corr_matrix = df[feature_columns].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    selected_features = [col for col in feature_columns if col not in to_drop]
    
    return selected_features

def analyze_extreme_weather_flags():
    """異常気象フラグの重要度を分析"""
    print("=== VT/VF 異常気象フラグ重要度分析 ===")
    
    # データ準備
    df = load_vtvf_data()
    df = prepare_features(df)
    feature_columns = select_features(df)
    
    print(f"✓ 初期特徴量数: {len(feature_columns)}")
    
    # データの準備
    X = df[feature_columns].copy()
    y = df['target']
    
    # NaN値の処理
    X = X.fillna(method='ffill').fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LightGBMで特徴量重要度を計算
    lgb_temp = lgb.LGBMClassifier(random_state=42)
    lgb_temp.fit(X_scaled, y)
    importances = lgb_temp.feature_importances_
    
    # 特徴量重要度をDataFrameに変換
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 異常気象フラグを抽出
    extreme_weather_flags = importance_df[importance_df['feature'].str.startswith('is_')]
    
    print(f"\n=== 異常気象フラグ重要度 ===")
    print(f"全特徴量数: {len(feature_columns)}")
    print(f"異常気象フラグ数: {len(extreme_weather_flags)}")
    
    if len(extreme_weather_flags) > 0:
        print("\n異常気象フラグの重要度ランキング:")
        for i, row in extreme_weather_flags.iterrows():
            rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
            print(f"{rank:3d}位: {row['feature']:<20} - {row['importance']:.4f}")
    else:
        print("異常気象フラグが見つかりませんでした")
    
    # 上位20位内の異常気象フラグを確認
    top_20 = importance_df.head(20)
    top_20_extreme_flags = top_20[top_20['feature'].str.startswith('is_')]
    
    print(f"\n=== 上位20位内の異常気象フラグ ===")
    if len(top_20_extreme_flags) > 0:
        for i, row in top_20_extreme_flags.iterrows():
            rank = top_20[top_20['feature'] == row['feature']].index[0] + 1
            print(f"{rank:2d}位: {row['feature']:<20} - {row['importance']:.4f}")
    else:
        print("上位20位内に異常気象フラグはありません")
    
    # 全体的な重要度の分布
    print(f"\n=== 重要度統計 ===")
    print(f"全特徴量の平均重要度: {importance_df['importance'].mean():.4f}")
    if len(extreme_weather_flags) > 0:
        print(f"異常気象フラグの平均重要度: {extreme_weather_flags['importance'].mean():.4f}")
        print(f"異常気象フラグの最高重要度: {extreme_weather_flags['importance'].max():.4f}")
        print(f"異常気象フラグの最低重要度: {extreme_weather_flags['importance'].min():.4f}")
    
    return importance_df, extreme_weather_flags

if __name__ == "__main__":
    importance_df, extreme_weather_flags = analyze_extreme_weather_flags() 