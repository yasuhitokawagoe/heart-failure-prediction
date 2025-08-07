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

def create_advanced_composite_features(df):
    """高度な複合気象特徴量を作成"""
    # 不快指数 (Discomfort Index)
    df['discomfort_index'] = 0.81 * df['avg_temp_vtvf'] + 0.01 * df['avg_humidity_vtvf'] * (0.99 * df['avg_temp_vtvf'] - 14.3) + 46.3
    
    # 体感温度 (Apparent Temperature) - 風速データがない場合は簡略化
    df['apparent_temp'] = df['avg_temp_vtvf'] + 0.348 * df['vapor_pressure_vtvf'] + 0.5
    
    # 熱ストレス指数 (Heat Stress Index)
    df['heat_stress_index'] = 0.5 * (df['avg_temp_vtvf'] + 61.0 + 1.2 * df['vapor_pressure_vtvf']) + 0.006 * df['vapor_pressure_vtvf'] * 2
    
    # 寒冷ストレス指数 (Cold Stress Index) - 風速データがない場合は簡略化
    df['cold_stress_index'] = 13.12 + 0.6215 * df['avg_temp_vtvf']
    
    # 気圧変化率
    df['pressure_change_rate'] = df['vapor_pressure_vtvf'].pct_change()
    
    # 温度-湿度-気圧複合指標
    df['temp_humidity_pressure'] = df['avg_temp_vtvf'] * df['avg_humidity_vtvf'] * df['vapor_pressure_vtvf'] / 1000
    
    # 気象変動指数
    df['weather_volatility'] = (
        df['avg_temp_vtvf'].rolling(7).std() + 
        df['avg_humidity_vtvf'].rolling(7).std() + 
        df['vapor_pressure_vtvf'].rolling(7).std()
    )
    
    # 気象ストレス指数
    df['weather_stress_index'] = (
        np.abs(df['avg_temp_vtvf'] - 22) * 0.5 +  # 22度からの偏差
        np.abs(df['avg_humidity_vtvf'] - 60) * 0.3 +  # 60%からの偏差
        np.abs(df['vapor_pressure_vtvf'] - df['vapor_pressure_vtvf'].rolling(30).mean()) * 0.2  # 気圧偏差
    )
    
    # 季節性気象ストレス
    df['seasonal_weather_stress'] = df['weather_stress_index'] * np.where(df['season'] == 'summer', 1.2, 
                                                                          np.where(df['season'] == 'winter', 1.1, 1.0))
    
    # 気象急変指数
    df['weather_change_index'] = (
        np.abs(df['avg_temp_vtvf'] - df['avg_temp_vtvf'].shift(1)) +
        np.abs(df['avg_humidity_vtvf'] - df['avg_humidity_vtvf'].shift(1)) * 0.5 +
        np.abs(df['vapor_pressure_vtvf'] - df['vapor_pressure_vtvf'].shift(1)) * 0.3
    )
    
    # 複合気象リスク指数
    df['composite_weather_risk'] = (
        df['discomfort_index'] * 0.3 +
        df['weather_stress_index'] * 0.3 +
        df['weather_change_index'] * 0.2 +
        df['weather_volatility'] * 0.2
    )
    
    # 気温-湿度相互作用指数
    df['temp_humidity_interaction'] = df['avg_temp_vtvf'] * df['avg_humidity_vtvf'] / 100
    
    # 気圧-温度相互作用指数
    df['pressure_temp_interaction'] = df['vapor_pressure_vtvf'] * df['avg_temp_vtvf'] / 100
    
    # 気象安定性指数
    df['weather_stability_index'] = 1 / (1 + df['weather_volatility'])
    
    # 気象リスク総合指数
    df['total_weather_risk'] = (
        df['discomfort_index'] * 0.25 +
        df['weather_stress_index'] * 0.25 +
        df['weather_change_index'] * 0.25 +
        df['weather_volatility'] * 0.25
    )
    
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
    df = create_advanced_composite_features(df)  # 複合特徴量を追加
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

def analyze_composite_weather_indicators():
    """複合気象指標の重要度を分析"""
    print("=== VT/VF 複合気象指標重要度分析 ===")
    
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
    
    # 複合気象指標を抽出
    composite_indicators = importance_df[importance_df['feature'].str.contains(
        'discomfort|apparent|stress|volatility|composite|weather_change|weather_stress|temp_humidity_pressure|pressure_change_rate'
    )]
    
    print(f"\n=== 複合気象指標重要度 ===")
    print(f"全特徴量数: {len(feature_columns)}")
    print(f"複合気象指標数: {len(composite_indicators)}")
    
    if len(composite_indicators) > 0:
        print("\n複合気象指標の重要度ランキング:")
        for i, row in composite_indicators.iterrows():
            rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
            print(f"{rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
    else:
        print("複合気象指標が見つかりませんでした")
    
    # 上位20位内の複合気象指標を確認
    top_20 = importance_df.head(20)
    top_20_composite = top_20[top_20['feature'].str.contains(
        'discomfort|apparent|stress|volatility|composite|weather_change|weather_stress|temp_humidity_pressure|pressure_change_rate'
    )]
    
    print(f"\n=== 上位20位内の複合気象指標 ===")
    if len(top_20_composite) > 0:
        for i, row in top_20_composite.iterrows():
            rank = top_20[top_20['feature'] == row['feature']].index[0] + 1
            print(f"{rank:2d}位: {row['feature']:<35} - {row['importance']:.4f}")
    else:
        print("上位20位内に複合気象指標はありません")
    
    # 複合気象指標の詳細分析
    print(f"\n=== 複合気象指標の詳細分析 ===")
    if len(composite_indicators) > 0:
        print("複合気象指標の種類別分析:")
        
        # 不快指数関連
        discomfort_features = composite_indicators[composite_indicators['feature'].str.contains('discomfort')]
        if len(discomfort_features) > 0:
            print(f"\n不快指数関連 ({len(discomfort_features)}個):")
            for i, row in discomfort_features.iterrows():
                rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
                print(f"  {rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
        
        # 体感温度関連
        apparent_features = composite_indicators[composite_indicators['feature'].str.contains('apparent')]
        if len(apparent_features) > 0:
            print(f"\n体感温度関連 ({len(apparent_features)}個):")
            for i, row in apparent_features.iterrows():
                rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
                print(f"  {rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
        
        # ストレス指数関連
        stress_features = composite_indicators[composite_indicators['feature'].str.contains('stress')]
        if len(stress_features) > 0:
            print(f"\nストレス指数関連 ({len(stress_features)}個):")
            for i, row in stress_features.iterrows():
                rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
                print(f"  {rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
        
        # 気象変動関連
        volatility_features = composite_indicators[composite_indicators['feature'].str.contains('volatility|change')]
        if len(volatility_features) > 0:
            print(f"\n気象変動関連 ({len(volatility_features)}個):")
            for i, row in volatility_features.iterrows():
                rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
                print(f"  {rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
        
        # 複合リスク関連
        composite_risk_features = composite_indicators[composite_indicators['feature'].str.contains('composite|temp_humidity_pressure')]
        if len(composite_risk_features) > 0:
            print(f"\n複合リスク関連 ({len(composite_risk_features)}個):")
            for i, row in composite_risk_features.iterrows():
                rank = importance_df[importance_df['feature'] == row['feature']].index[0] + 1
                print(f"  {rank:3d}位: {row['feature']:<35} - {row['importance']:.4f}")
    
    # 全体的な重要度の分布
    print(f"\n=== 重要度統計 ===")
    print(f"全特徴量の平均重要度: {importance_df['importance'].mean():.4f}")
    if len(composite_indicators) > 0:
        print(f"複合気象指標の平均重要度: {composite_indicators['importance'].mean():.4f}")
        print(f"複合気象指標の最高重要度: {composite_indicators['importance'].max():.4f}")
        print(f"複合気象指標の最低重要度: {composite_indicators['importance'].min():.4f}")
    
    return importance_df, composite_indicators

if __name__ == "__main__":
    importance_df, composite_indicators = analyze_composite_weather_indicators() 