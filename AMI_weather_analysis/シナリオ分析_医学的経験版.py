import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def get_saved_model_features():
    """保存されたモデルの特徴量セットを取得"""
    feature_importance_df = pd.read_csv('保存モデル/結果/feature_importance.csv')
    saved_features = feature_importance_df['feature'].tolist()
    return saved_features

def encode_simplified_weather_conditions(df):
    """天気概況を6分類に簡略化して数値化"""
    weather_mapping = {
        '晴れ': '晴れ系', '快晴': '晴れ系', '曇り': '曇り系', '薄曇': '曇り系',
        '小雨': '小雨', '大雨': '大雨', '雷雨': '雷雨', '雪': '雪'
    }
    df['weather_simplified'] = df['天気分類(統合)'].map(weather_mapping)
    df['weather_simplified'] = df['weather_simplified'].fillna('曇り系')
    le = LabelEncoder()
    df['weather_simplified_encoded'] = le.fit_transform(df['weather_simplified'])
    return df, le

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    import jpholiday
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
    """気象要素の相互作用特徴量を作成"""
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['vapor_pressure']
    df['humidity_pressure'] = df['avg_humidity'] * df['vapor_pressure']
    
    df['temp_change'] = df['avg_temp'].diff()
    df['humidity_change'] = df['avg_humidity'].diff()
    df['pressure_change'] = df['vapor_pressure'].diff()
    
    df['temp_change_humidity_change'] = df['temp_change'] * df['humidity_change']
    df['temp_change_pressure_change'] = df['temp_change'] * df['pressure_change']
    df['humidity_change_pressure_change'] = df['humidity_change'] * df['pressure_change']
    
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    return df

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    
    for col in weather_cols:
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].rolling(window=window, min_periods=window).std()
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
    
    return df

def create_seasonal_weighted_features(df):
    """季節性を考慮した重み付け特徴量を作成"""
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def create_additional_features(df):
    """保存されたモデルに必要な追加特徴量を作成"""
    # 月次変化率
    df['avg_temp_monthly_change_rate'] = df['avg_temp'].pct_change(30)
    df['avg_humidity_monthly_change_rate'] = df['avg_humidity'].pct_change(30)
    
    # 週次変化率
    df['vapor_pressure_weekly_change_rate'] = df['vapor_pressure'].pct_change(7)
    df['avg_humidity_weekly_change_rate'] = df['avg_humidity'].pct_change(7)
    
    # 季節性偏差
    df['temp_seasonal_deviation_7d'] = df['avg_temp'] - df['avg_temp'].rolling(7).mean()
    df['temp_seasonal_deviation_14d'] = df['avg_temp'] - df['avg_temp'].rolling(14).mean()
    df['seasonal_temp_deviation'] = df['avg_temp'] - df.groupby('month')['avg_temp'].transform('mean')
    
    # トレンド強度
    df['avg_temp_trend_strength'] = df['avg_temp'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['avg_humidity_trend_strength'] = df['avg_humidity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['vapor_pressure_trend_strength'] = df['vapor_pressure'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    # トレンドスロープ
    df['avg_humidity_trend_slope'] = df['avg_humidity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    # 3日変化
    df['temp_change_3d'] = df['avg_temp'].diff(3)
    
    # 異常スコア
    df['avg_temp_anomaly_score'] = (df['avg_temp'] - df['avg_temp'].rolling(30).mean()) / df['avg_temp'].rolling(30).std()
    df['avg_humidity_anomaly_score'] = (df['avg_humidity'] - df['avg_humidity'].rolling(30).mean()) / df['avg_humidity'].rolling(30).std()
    
    # 予測可能性
    df['avg_temp_predictability'] = df['avg_temp'].rolling(7).std() / df['avg_temp'].rolling(30).std()
    
    # 加速度
    df['avg_temp_acceleration'] = df['avg_temp'].diff().diff()
    df['avg_temp_acceleration_2nd'] = df['avg_temp'].diff().diff().diff()
    
    # 気圧変化と温度変化の相互作用
    df['pressure_change_temp_change'] = df['pressure_change'] * df['temp_change']
    
    return df

def load_model_and_data():
    """保存されたモデルとデータを読み込み"""
    try:
        model = joblib.load('保存モデル/AMI予測モデル_天気概況6分類版_XGBoost.pkl')
        
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people']
        
        return model, df
    except Exception as e:
        print(f"エラー: {e}")
        return None, None

def prepare_data_with_saved_features(df, saved_features):
    """保存されたモデルの特徴量セットでデータを準備"""
    # 日付でソート
    df = df.sort_values('hospitalization_date')
    
    # 天気概況を6分類に簡略化
    df, weather_encoder = encode_simplified_weather_conditions(df)
    
    # ターゲット変数の作成
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    # 特徴量エンジニアリング
    df = create_date_features(df)
    df = create_weather_interaction_features(df)
    df = create_time_series_features(df)
    df = create_advanced_weather_timeseries_features(df)
    df = create_seasonal_weighted_features(df)
    df = create_additional_features(df)
    
    # 保存されたモデルの特徴量セットを使用
    available_features = [f for f in saved_features if f in df.columns]
    
    # データを準備
    X = df[available_features].values
    y = df['target'].values
    
    # 欠損値の処理
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, available_features, df, scaler

def create_typhoon_scenario_analysis(model, df, feature_names, scaler):
    """台風シナリオ分析"""
    print("=== 台風シナリオ分析 ===")
    
    # 台風関連の条件を定義
    typhoon_conditions = [
        {'avg_wind': 15, 'avg_humidity': 90, 'vapor_pressure': 25, 'is_holiday': 0},  # 強風・高湿度
        {'avg_wind': 20, 'avg_humidity': 95, 'vapor_pressure': 30, 'is_holiday': 0},  # 暴風・極高湿度
        {'avg_wind': 25, 'avg_humidity': 98, 'vapor_pressure': 35, 'is_holiday': 1},  # 暴風・休日
    ]
    
    results = []
    for i, condition in enumerate(typhoon_conditions):
        # ベースライン条件を作成
        base_condition = df[feature_names].iloc[0].copy()
        
        # 台風条件を適用
        for key, value in condition.items():
            if key in feature_names:
                base_condition[key] = value
        
        # 特徴量エンジニアリングを適用
        scenario_df = df.iloc[0:1].copy()
        for key, value in condition.items():
            if key in scenario_df.columns:
                scenario_df[key] = value
        
        # 特徴量を再計算
        scenario_df = create_date_features(scenario_df)
        scenario_df = create_weather_interaction_features(scenario_df)
        scenario_df = create_time_series_features(scenario_df)
        scenario_df = create_advanced_weather_timeseries_features(scenario_df)
        scenario_df = create_seasonal_weighted_features(scenario_df)
        scenario_df = create_additional_features(scenario_df)
        
        # 予測用データを準備
        X_scenario = scenario_df[feature_names].values
        X_scenario = np.nan_to_num(X_scenario, nan=0.0, posinf=0.0, neginf=0.0)
        X_scenario_scaled = scaler.transform(X_scenario)
        
        # 予測
        risk_prob = model.predict_proba(X_scenario_scaled)[0][1]
        
        results.append({
            'scenario': f'台風シナリオ{i+1}',
            'conditions': condition,
            'risk_probability': risk_prob,
            'risk_level': '高リスク' if risk_prob > 0.5 else '低リスク'
        })
        
        print(f"台風シナリオ{i+1}: リスク確率 {risk_prob:.3f} ({'高リスク' if risk_prob > 0.5 else '低リスク'})")
    
    return results

def create_heavy_rain_scenario_analysis(model, df, feature_names, scaler):
    """集中豪雨シナリオ分析"""
    print("=== 集中豪雨シナリオ分析 ===")
    
    # 集中豪雨関連の条件を定義
    heavy_rain_conditions = [
        {'avg_humidity': 95, 'vapor_pressure': 28, 'avg_temp': 25, 'is_holiday': 0},  # 高湿度・適温
        {'avg_humidity': 98, 'vapor_pressure': 32, 'avg_temp': 28, 'is_holiday': 1},  # 極高湿度・高温・休日
        {'avg_humidity': 92, 'vapor_pressure': 26, 'avg_temp': 22, 'is_holiday': 0},  # 高湿度・低温
    ]
    
    results = []
    for i, condition in enumerate(heavy_rain_conditions):
        # ベースライン条件を作成
        base_condition = df[feature_names].iloc[0].copy()
        
        # 集中豪雨条件を適用
        for key, value in condition.items():
            if key in feature_names:
                base_condition[key] = value
        
        # 特徴量エンジニアリングを適用
        scenario_df = df.iloc[0:1].copy()
        for key, value in condition.items():
            if key in scenario_df.columns:
                scenario_df[key] = value
        
        # 特徴量を再計算
        scenario_df = create_date_features(scenario_df)
        scenario_df = create_weather_interaction_features(scenario_df)
        scenario_df = create_time_series_features(scenario_df)
        scenario_df = create_advanced_weather_timeseries_features(scenario_df)
        scenario_df = create_seasonal_weighted_features(scenario_df)
        scenario_df = create_additional_features(scenario_df)
        
        # 予測用データを準備
        X_scenario = scenario_df[feature_names].values
        X_scenario = np.nan_to_num(X_scenario, nan=0.0, posinf=0.0, neginf=0.0)
        X_scenario_scaled = scaler.transform(X_scenario)
        
        # 予測
        risk_prob = model.predict_proba(X_scenario_scaled)[0][1]
        
        results.append({
            'scenario': f'集中豪雨シナリオ{i+1}',
            'conditions': condition,
            'risk_probability': risk_prob,
            'risk_level': '高リスク' if risk_prob > 0.5 else '低リスク'
        })
        
        print(f"集中豪雨シナリオ{i+1}: リスク確率 {risk_prob:.3f} ({'高リスク' if risk_prob > 0.5 else '低リスク'})")
    
    return results

def create_seasonal_change_scenario_analysis(model, df, feature_names, scaler):
    """季節変化シナリオ分析"""
    print("=== 季節変化シナリオ分析 ===")
    
    # 季節変化関連の条件を定義
    seasonal_conditions = [
        {'month': 1, 'avg_temp': 5, 'avg_humidity': 60, 'is_holiday': 1},   # 冬・寒い・休日
        {'month': 7, 'avg_temp': 30, 'avg_humidity': 80, 'is_holiday': 0},  # 夏・暑い・平日
        {'month': 4, 'avg_temp': 15, 'avg_humidity': 70, 'is_holiday': 1},  # 春・適温・休日
        {'month': 10, 'avg_temp': 20, 'avg_humidity': 65, 'is_holiday': 0}, # 秋・適温・平日
    ]
    
    results = []
    for i, condition in enumerate(seasonal_conditions):
        # ベースライン条件を作成
        base_condition = df[feature_names].iloc[0].copy()
        
        # 季節変化条件を適用
        for key, value in condition.items():
            if key in feature_names:
                base_condition[key] = value
        
        # 特徴量エンジニアリングを適用
        scenario_df = df.iloc[0:1].copy()
        for key, value in condition.items():
            if key in scenario_df.columns:
                scenario_df[key] = value
        
        # 特徴量を再計算
        scenario_df = create_date_features(scenario_df)
        scenario_df = create_weather_interaction_features(scenario_df)
        scenario_df = create_time_series_features(scenario_df)
        scenario_df = create_advanced_weather_timeseries_features(scenario_df)
        scenario_df = create_seasonal_weighted_features(scenario_df)
        scenario_df = create_additional_features(scenario_df)
        
        # 予測用データを準備
        X_scenario = scenario_df[feature_names].values
        X_scenario = np.nan_to_num(X_scenario, nan=0.0, posinf=0.0, neginf=0.0)
        X_scenario_scaled = scaler.transform(X_scenario)
        
        # 予測
        risk_prob = model.predict_proba(X_scenario_scaled)[0][1]
        
        results.append({
            'scenario': f'季節変化シナリオ{i+1}',
            'conditions': condition,
            'risk_probability': risk_prob,
            'risk_level': '高リスク' if risk_prob > 0.5 else '低リスク'
        })
        
        season_name = {1: '冬', 7: '夏', 4: '春', 10: '秋'}[condition['month']]
        print(f"{season_name}シナリオ: リスク確率 {risk_prob:.3f} ({'高リスク' if risk_prob > 0.5 else '低リスク'})")
    
    return results

def create_long_holiday_scenario_analysis(model, df, feature_names, scaler):
    """長期休暇シナリオ分析"""
    print("=== 長期休暇シナリオ分析 ===")
    
    # 長期休暇関連の条件を定義
    holiday_conditions = [
        {'is_holiday': 1, 'avg_temp': 25, 'avg_humidity': 70, 'month': 8},  # 夏休み
        {'is_holiday': 1, 'avg_temp': 10, 'avg_humidity': 60, 'month': 12}, # 年末年始
        {'is_holiday': 1, 'avg_temp': 20, 'avg_humidity': 65, 'month': 5},  # ゴールデンウィーク
        {'is_holiday': 0, 'avg_temp': 15, 'avg_humidity': 70, 'month': 3},  # 平日・春
    ]
    
    results = []
    for i, condition in enumerate(holiday_conditions):
        # ベースライン条件を作成
        base_condition = df[feature_names].iloc[0].copy()
        
        # 長期休暇条件を適用
        for key, value in condition.items():
            if key in feature_names:
                base_condition[key] = value
        
        # 特徴量エンジニアリングを適用
        scenario_df = df.iloc[0:1].copy()
        for key, value in condition.items():
            if key in scenario_df.columns:
                scenario_df[key] = value
        
        # 特徴量を再計算
        scenario_df = create_date_features(scenario_df)
        scenario_df = create_weather_interaction_features(scenario_df)
        scenario_df = create_time_series_features(scenario_df)
        scenario_df = create_advanced_weather_timeseries_features(scenario_df)
        scenario_df = create_seasonal_weighted_features(scenario_df)
        scenario_df = create_additional_features(scenario_df)
        
        # 予測用データを準備
        X_scenario = scenario_df[feature_names].values
        X_scenario = np.nan_to_num(X_scenario, nan=0.0, posinf=0.0, neginf=0.0)
        X_scenario_scaled = scaler.transform(X_scenario)
        
        # 予測
        risk_prob = model.predict_proba(X_scenario_scaled)[0][1]
        
        results.append({
            'scenario': f'長期休暇シナリオ{i+1}',
            'conditions': condition,
            'risk_probability': risk_prob,
            'risk_level': '高リスク' if risk_prob > 0.5 else '低リスク'
        })
        
        holiday_name = {8: '夏休み', 12: '年末年始', 5: 'ゴールデンウィーク', 3: '平日'}[condition['month']]
        print(f"{holiday_name}シナリオ: リスク確率 {risk_prob:.3f} ({'高リスク' if risk_prob > 0.5 else '低リスク'})")
    
    return results

def create_scenario_comparison_visualization(all_results):
    """シナリオ比較の可視化"""
    print("=== シナリオ比較の可視化 ===")
    
    # データを整理
    scenarios = []
    risk_probs = []
    categories = []
    
    for category, results in all_results.items():
        for result in results:
            scenarios.append(result['scenario'])
            risk_probs.append(result['risk_probability'])
            categories.append(category)
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # カテゴリ別に色分け
    colors = {'台風': 'red', '集中豪雨': 'blue', '季節変化': 'green', '長期休暇': 'orange'}
    
    for category in colors.keys():
        cat_data = [(s, p) for s, p, c in zip(scenarios, risk_probs, categories) if c == category]
        if cat_data:
            scenarios_cat, probs_cat = zip(*cat_data)
            plt.bar(scenarios_cat, probs_cat, color=colors[category], alpha=0.7, label=category)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='高リスク閾値')
    plt.xlabel('シナリオ', fontsize=12)
    plt.ylabel('リスク確率', fontsize=12)
    plt.title('シナリオ別AMIリスク分析', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('シナリオ分析結果/scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("シナリオ比較グラフを保存しました")

def create_detailed_scenario_report(all_results):
    """詳細なシナリオ分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    with open('シナリオ分析結果/シナリオ分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# シナリオ分析レポート\n\n')
        
        f.write('## 概要\n')
        f.write('- 医学的経験に基づく具体的な気象シナリオ分析\n')
        f.write('- 台風、集中豪雨、季節変化、長期休暇の影響評価\n')
        f.write('- 予防医療におけるリスク予測の実用化\n\n')
        
        f.write('## シナリオ別分析結果\n\n')
        
        for category, results in all_results.items():
            f.write(f'### {category}\n\n')
            
            for result in results:
                f.write(f'#### {result["scenario"]}\n')
                f.write(f'- リスク確率: {result["risk_probability"]:.3f}\n')
                f.write(f'- リスクレベル: {result["risk_level"]}\n')
                f.write(f'- 条件: {result["conditions"]}\n\n')
        
        f.write('## 医学的解釈\n\n')
        f.write('### 1. 台風の影響\n')
        f.write('- 強風によるストレス増加\n')
        f.write('- 気圧変化による循環器系への負荷\n')
        f.write('- 避難行動による身体活動の変化\n\n')
        
        f.write('### 2. 集中豪雨の影響\n')
        f.write('- 高湿度による体温調節機能への負荷\n')
        f.write('- 気圧変化による自律神経系への影響\n')
        f.write('- 外出制限による運動不足\n\n')
        
        f.write('### 3. 季節変化の影響\n')
        f.write('- 温度変化による血管収縮・拡張\n')
        f.write('- 季節性うつ病との関連\n')
        f.write('- 生活習慣の季節変動\n\n')
        
        f.write('### 4. 長期休暇の影響\n')
        f.write('- 生活リズムの変化\n')
        f.write('- 食事・運動パターンの変化\n')
        f.write('- ストレスレベルの変動\n\n')
        
        f.write('## 予防医療への応用\n')
        f.write('- 気象予報に基づくリスク予測\n')
        f.write('- 高リスク日の事前警告システム\n')
        f.write('- 個別化された予防指導\n')
        f.write('- 医療資源の効率的配分\n')
    
    print("詳細レポートを保存しました: シナリオ分析結果/シナリオ分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        import os
        os.makedirs('シナリオ分析結果', exist_ok=True)
        
        print("保存されたモデルの特徴量セットを取得中...")
        saved_features = get_saved_model_features()
        
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        print("データを準備中...")
        X_scaled, y, feature_names, df, scaler = prepare_data_with_saved_features(df, saved_features)
        
        print("シナリオ分析を実行中...")
        
        # 1. 台風シナリオ分析
        typhoon_results = create_typhoon_scenario_analysis(model, df, feature_names, scaler)
        
        # 2. 集中豪雨シナリオ分析
        heavy_rain_results = create_heavy_rain_scenario_analysis(model, df, feature_names, scaler)
        
        # 3. 季節変化シナリオ分析
        seasonal_results = create_seasonal_change_scenario_analysis(model, df, feature_names, scaler)
        
        # 4. 長期休暇シナリオ分析
        holiday_results = create_long_holiday_scenario_analysis(model, df, feature_names, scaler)
        
        # 5. シナリオ比較の可視化
        all_results = {
            '台風': typhoon_results,
            '集中豪雨': heavy_rain_results,
            '季節変化': seasonal_results,
            '長期休暇': holiday_results
        }
        create_scenario_comparison_visualization(all_results)
        
        # 6. 詳細レポート
        create_detailed_scenario_report(all_results)
        
        print("\n=== シナリオ分析完了 ===")
        print("結果は 'シナリオ分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 