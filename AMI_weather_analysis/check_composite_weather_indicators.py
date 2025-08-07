import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def check_composite_weather_indicators():
    """気候の複合指標の重要度を確認"""
    print("=== 気候の複合指標の重要度確認 ===")
    
    # データの読み込み
    df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
    df['hospitalization_date'] = pd.to_datetime(df['date'])
    df['hospitalization_count'] = df['people']
    
    # 特徴量の準備（SHAP解析と同じ処理）
    from datetime import datetime, timedelta
    import jpholiday
    
    # 日付関連の特徴量
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性指標
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
    
    # 月末・月初フラグ
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end.astype(int)
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    # 異常気象フラグ
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    df['is_summer_day'] = (df['max_temp'] >= 25).astype(int)
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    
    # 気象相互作用（複合指標）
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['vapor_pressure']
    df['temp_change'] = (df['avg_temp'] - df['avg_temp'].shift(1)).fillna(0)
    df['humidity_change'] = (df['avg_humidity'] - df['avg_humidity'].shift(1)).fillna(0)
    df['pressure_change'] = (df['vapor_pressure'] - df['vapor_pressure'].shift(1)).fillna(0)
    
    # 不快指数（複合指標）
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    # 追加の複合指標
    # 体感温度（風速考慮）
    df['apparent_temp'] = df['avg_temp'] + 0.348 * df['vapor_pressure'] - 0.7 * df['avg_wind'] + 0.7 * (df['avg_wind'] / (1.76 + 0.75 * df['avg_wind'])) - 4.25
    
    # 熱ストレス指数
    df['heat_stress_index'] = 0.5 * (df['avg_temp'] + 61.0 + 1.2 * df['vapor_pressure'] - 0.006 * df['vapor_pressure'] * df['avg_temp'])
    
    # 寒冷ストレス指数
    df['cold_stress_index'] = 13.12 + 0.6215 * df['avg_temp'] - 11.37 * (df['avg_wind'] ** 0.16) + 0.3965 * df['avg_temp'] * (df['avg_wind'] ** 0.16)
    
    # 気圧変化率
    df['pressure_change_rate'] = df['vapor_pressure'].pct_change()
    
    # 温度湿度圧力の複合指標
    df['temp_humidity_pressure'] = df['avg_temp'] * df['avg_humidity'] * df['vapor_pressure']
    
    # 時系列特徴量
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    # 高度な時系列特徴量
    for col in weather_cols:
        # ラグ特徴量
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        # 変化率
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        # 移動平均の変化率
        for window in [3, 7]:
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
    
    # 季節性重み付け
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    # 季節性重み付けされた気象特徴量
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    # ターゲット変数
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    # 特徴量の選択
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 
                   'hospitalization_count', 'people']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
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
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    print(f"✓ 全特徴量数: {len(feature_columns)}")
    
    # 複合指標の特定
    composite_indicators = [
        'temp_humidity', 'temp_pressure', 'discomfort_index', 'apparent_temp',
        'heat_stress_index', 'cold_stress_index', 'temp_humidity_pressure',
        'pressure_change_rate'
    ]
    
    # 実際に存在する複合指標を確認
    existing_composite_indicators = [col for col in composite_indicators if col in feature_columns]
    print(f"\n=== 複合指標一覧 ===")
    for i, indicator in enumerate(existing_composite_indicators, 1):
        print(f"{i:2d}. {indicator}")
    
    # モデルの読み込み
    model = joblib.load('saved_models/xgb_model_latest.pkl')
    
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
    import lightgbm as lgb
    lgb_temp = lgb.LGBMClassifier(random_state=42)
    lgb_temp.fit(X_scaled, y)
    importances = lgb_temp.feature_importances_
    
    # 上位100個の特徴量を選択
    top_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])][:100]
    
    # 高相関な特徴量を除去
    X_df = pd.DataFrame(X_scaled, columns=feature_columns)
    reduced_features = remove_highly_correlated_features(X_df, top_features)
    
    # 65個に調整
    if len(reduced_features) > 65:
        sorted_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]) if f in reduced_features]
        reduced_features = sorted_features[:65]
    elif len(reduced_features) < 65:
        sorted_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])]
        additional_features = [f for f in sorted_features if f not in reduced_features][:65-len(reduced_features)]
        reduced_features.extend(additional_features)
    
    print(f"\n=== 選択された特徴量（65個） ===")
    for i, feature in enumerate(reduced_features, 1):
        print(f"{i:2d}. {feature}")
    
    # 複合指標の重要度確認
    print(f"\n=== 複合指標の重要度 ===")
    composite_indicators_in_selected = [f for f in reduced_features if f in existing_composite_indicators]
    for indicator in composite_indicators_in_selected:
        if indicator in feature_columns:
            idx = feature_columns.index(indicator)
            importance = importances[idx]
            rank = list(importances).index(importance) + 1
            print(f"{indicator}: {importance:.4f} (順位: {rank}/{len(importances)})")
    
    # 全複合指標の重要度
    print(f"\n=== 全複合指標の重要度 ===")
    for indicator in existing_composite_indicators:
        if indicator in feature_columns:
            idx = feature_columns.index(indicator)
            importance = importances[idx]
            rank = list(importances).index(importance) + 1
            print(f"{indicator}: {importance:.4f} (順位: {rank}/{len(importances)})")
    
    # 複合指標の詳細分析
    print(f"\n=== 複合指標の詳細分析 ===")
    print("1. 気象相互作用指標:")
    interaction_indicators = ['temp_humidity', 'temp_pressure', 'temp_humidity_pressure']
    for indicator in interaction_indicators:
        if indicator in feature_columns:
            idx = feature_columns.index(indicator)
            importance = importances[idx]
            rank = list(importances).index(importance) + 1
            print(f"   {indicator}: {importance:.4f} (順位: {rank}/{len(importances)})")
    
    print("\n2. 体感指標:")
    comfort_indicators = ['discomfort_index', 'apparent_temp', 'heat_stress_index', 'cold_stress_index']
    for indicator in comfort_indicators:
        if indicator in feature_columns:
            idx = feature_columns.index(indicator)
            importance = importances[idx]
            rank = list(importances).index(importance) + 1
            print(f"   {indicator}: {importance:.4f} (順位: {rank}/{len(importances)})")
    
    print("\n3. 変化率指標:")
    change_indicators = ['pressure_change_rate']
    for indicator in change_indicators:
        if indicator in feature_columns:
            idx = feature_columns.index(indicator)
            importance = importances[idx]
            rank = list(importances).index(importance) + 1
            print(f"   {indicator}: {importance:.4f} (順位: {rank}/{len(importances)})")

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
    check_composite_weather_indicators() 