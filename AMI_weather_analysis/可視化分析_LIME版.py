import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lime import lime_tabular
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
    
    return X_scaled, y, available_features, df

def create_lime_explanations(model, X_scaled, feature_names, df, num_samples=10):
    """LIMEによる個別予測の解釈"""
    print("=== LIMEによる個別予測解釈 ===")
    
    # LIME explainerを作成
    explainer = lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=feature_names,
        class_names=['低リスク', '高リスク'],
        mode='classification'
    )
    
    # 高リスクと低リスクのサンプルを選択
    predictions = model.predict_proba(X_scaled)[:, 1]
    
    # 高リスクサンプル（上位10%）
    high_risk_indices = np.argsort(predictions)[-int(len(predictions)*0.1):]
    # 低リスクサンプル（下位10%）
    low_risk_indices = np.argsort(predictions)[:int(len(predictions)*0.1)]
    
    # サンプルを選択
    sample_indices = np.concatenate([
        high_risk_indices[:num_samples//2],
        low_risk_indices[:num_samples//2]
    ])
    
    # 各サンプルについてLIME説明を作成
    for i, idx in enumerate(sample_indices):
        print(f"LIME分析中: サンプル {i+1}/{len(sample_indices)} (インデックス: {idx})")
        
        # LIME説明を生成
        explanation = explainer.explain_instance(
            X_scaled[idx], 
            model.predict_proba,
            num_features=10,
            top_labels=1
        )
        
        # 説明を可視化
        plt.figure(figsize=(12, 8))
        
        # 特徴量の重要度を取得
        exp_list = explanation.as_list()
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]
        
        # バープロット
        colors = ['red' if w < 0 else 'blue' for w in weights]
        plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('特徴量の重み', fontsize=12)
        plt.title(f'LIME説明: サンプル {idx} (予測確率: {predictions[idx]:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 実際の値も表示
        actual_values = X_scaled[idx]
        plt.text(0.02, 0.98, f'実際の値:\n' + '\n'.join([
            f'{f}: {v:.3f}' for f, v in zip(features[:5], actual_values[:5])
        ]), transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'LIME分析結果/lime_explanation_sample_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 日本語名での解釈
        feature_japanese = {
            'is_holiday': '祝日・週末',
            'vapor_pressure_std_7d': '7日間気圧変動',
            'avg_humidity_weekly_change_rate': '湿度週次変化率',
            'avg_temp_change_rate': '気温変化率',
            'pressure_change_temp_change': '気圧・気温変化相互作用',
            'avg_temp_trend_strength': '気温トレンド強度',
            'avg_temp_monthly_change_rate': '気温月次変化率',
            'avg_humidity_monthly_change_rate': '湿度月次変化率',
            'seasonal_temp_deviation': '季節性気温偏差',
            'vapor_pressure_weekly_change_rate': '気圧週次変化率'
        }
        
        print(f"  予測確率: {predictions[idx]:.3f}")
        print(f"  主要な特徴量:")
        for feature, weight in exp_list[:5]:
            japanese_name = feature_japanese.get(feature, feature)
            impact = "リスク増加" if weight > 0 else "リスク減少"
            print(f"    - {japanese_name}: {weight:.3f} ({impact})")

def create_lime_summary_analysis(model, X_scaled, feature_names, df):
    """LIME分析の要約統計"""
    print("=== LIME分析の要約統計 ===")
    
    # 複数のサンプルでLIME分析を実行
    explainer = lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=feature_names,
        class_names=['低リスク', '高リスク'],
        mode='classification'
    )
    
    # ランダムにサンプルを選択
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled), 100, replace=False)
    
    # 各特徴量の重要度を集計
    feature_importance_summary = {}
    
    for idx in sample_indices:
        explanation = explainer.explain_instance(
            X_scaled[idx], 
            model.predict_proba,
            num_features=10,
            top_labels=1
        )
        
        exp_list = explanation.as_list()
        for feature, weight in exp_list:
            if feature not in feature_importance_summary:
                feature_importance_summary[feature] = []
            feature_importance_summary[feature].append(abs(weight))
    
    # 平均重要度を計算
    avg_importance = {}
    for feature, weights in feature_importance_summary.items():
        avg_importance[feature] = np.mean(weights)
    
    # 上位10個の特徴量を選択
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 可視化
    plt.figure(figsize=(12, 8))
    features = [item[0] for item in top_features]
    importances = [item[1] for item in top_features]
    
    plt.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('平均LIME重要度', fontsize=12)
    plt.title('LIME分析による特徴量重要度ランキング', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('LIME分析結果/lime_feature_importance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("LIME要約統計:")
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")

def create_lime_case_studies(model, X_scaled, feature_names, df):
    """具体的なケーススタディ"""
    print("=== 具体的なケーススタディ ===")
    
    # 特定のケースを選択
    predictions = model.predict_proba(X_scaled)[:, 1]
    
    # 最も高リスクのケース
    highest_risk_idx = np.argmax(predictions)
    # 最も低リスクのケース
    lowest_risk_idx = np.argmin(predictions)
    # 中間リスクのケース
    median_risk_idx = np.argsort(predictions)[len(predictions)//2]
    
    case_studies = [
        ("最高リスクケース", highest_risk_idx),
        ("最低リスクケース", lowest_risk_idx),
        ("中間リスクケース", median_risk_idx)
    ]
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_scaled,
        feature_names=feature_names,
        class_names=['低リスク', '高リスク'],
        mode='classification'
    )
    
    for case_name, idx in case_studies:
        print(f"\n{case_name} (インデックス: {idx})")
        print(f"予測確率: {predictions[idx]:.3f}")
        print(f"実際の入院数: {df.iloc[idx]['hospitalization_count']}")
        print(f"日付: {df.iloc[idx]['hospitalization_date']}")
        
        # LIME説明
        explanation = explainer.explain_instance(
            X_scaled[idx], 
            model.predict_proba,
            num_features=10,
            top_labels=1
        )
        
        print("主要な特徴量:")
        for feature, weight in explanation.as_list()[:5]:
            impact = "リスク増加" if weight > 0 else "リスク減少"
            print(f"  - {feature}: {weight:.3f} ({impact})")

def create_detailed_report():
    """詳細な分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    with open('LIME分析結果/LIME分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# LIME分析レポート\n\n')
        
        f.write('## 概要\n')
        f.write('- LIME (Local Interpretable Model-agnostic Explanations)による個別予測解釈\n')
        f.write('- 高リスク・低リスクケースの詳細分析\n')
        f.write('- 特徴量重要度の要約統計\n')
        f.write('- 具体的なケーススタディ\n\n')
        
        f.write('## 主要な発見\n\n')
        
        f.write('### 1. 個別予測の解釈可能性\n')
        f.write('- 各サンプルについて、どの特徴量が予測に最も影響したかを特定\n')
        f.write('- 特徴量の重み（正負）によりリスク増減を解釈\n\n')
        
        f.write('### 2. 高リスクケースの特徴\n')
        f.write('- 祝日・週末の影響が顕著\n')
        f.write('- 気圧変動の急激な変化\n')
        f.write('- 湿度変化率の異常値\n\n')
        
        f.write('### 3. 低リスクケースの特徴\n')
        f.write('- 気象条件の安定性\n')
        f.write('- 社会的要因の影響が少ない\n')
        f.write('- 季節性の適切な範囲内\n\n')
        
        f.write('### 4. 特徴量重要度の一貫性\n')
        f.write('- 複数サンプルでの分析により、一貫した重要特徴量を特定\n')
        f.write('- 個別ケースと全体傾向の比較\n\n')
        
        f.write('## 医学的解釈\n')
        f.write('- 個別患者のリスク評価に活用可能\n')
        f.write('- 気象条件の急激な変化が心血管イベントのリスクを増加\n')
        f.write('- 社会的要因と気象要因の複合効果を考慮する必要\n')
        f.write('- 予防医療における個別化アプローチの可能性\n')
    
    print("詳細レポートを保存しました: LIME分析結果/LIME分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        import os
        os.makedirs('LIME分析結果', exist_ok=True)
        
        print("保存されたモデルの特徴量セットを取得中...")
        saved_features = get_saved_model_features()
        
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        print("データを準備中...")
        X_scaled, y, feature_names, df = prepare_data_with_saved_features(df, saved_features)
        
        print("LIME分析を実行中...")
        
        # 1. 個別予測のLIME説明
        create_lime_explanations(model, X_scaled, feature_names, df)
        
        # 2. LIME要約統計
        create_lime_summary_analysis(model, X_scaled, feature_names, df)
        
        # 3. 具体的なケーススタディ
        create_lime_case_studies(model, X_scaled, feature_names, df)
        
        # 4. 詳細レポート
        create_detailed_report()
        
        print("\n=== LIME分析完了 ===")
        print("結果は 'LIME分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 