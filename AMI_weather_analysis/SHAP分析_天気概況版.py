import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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
    import jpholiday
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性指標の強化
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
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

def create_weather_interaction_features(df):
    """気象要素の相互作用特徴量を作成"""
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
    
    return df

def create_seasonal_weighted_features(df):
    """季節性を考慮した重み付け特徴量を作成"""
    # 季節性重み
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def load_model_and_data():
    """保存されたモデルとデータを読み込み"""
    try:
        # モデルの読み込み
        model = joblib.load('保存モデル/AMI予測モデル_天気概況6分類版_XGBoost.pkl')
        
        # データの読み込み
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people']
        
        return model, df
    except Exception as e:
        print(f"エラー: {e}")
        return None, None

def prepare_data_for_shap(df):
    """SHAP分析用のデータを準備"""
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
    
    # 特徴量の選択
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 'hospitalization_count', 'people']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院人数関連の特徴量を除外
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
    
    # 欠損値の処理
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df, numeric_cols, weather_encoder

def perform_shap_analysis(model, df, feature_columns):
    """SHAP分析を実行"""
    print("=== SHAP分析を開始 ===")
    
    # データの準備
    X = df[feature_columns].values
    y = df['target'].values
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SHAP値を計算
    print("SHAP値を計算中...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # 特徴量名を取得
    feature_names = feature_columns
    
    return shap_values, X_scaled, y, feature_names, explainer

def create_shap_plots(shap_values, X_scaled, y, feature_names, explainer):
    """SHAPプロットを作成"""
    print("=== SHAPプロットを作成 ===")
    
    # 1. 概要プロット（全体的な特徴量重要度）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - 全特徴量の重要度', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('SHAP分析結果/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 天気概況関連の特徴量のみ
    weather_features = [f for f in feature_names if 'weather' in f.lower() or 'encoded' in f.lower()]
    if weather_features:
        weather_indices = [feature_names.index(f) for f in weather_features]
        weather_shap = shap_values[:, weather_indices]
        weather_X = X_scaled[:, weather_indices]
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(weather_shap, weather_X, feature_names=weather_features, show=False)
        plt.title('SHAP Summary Plot - 天気概況関連特徴量', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('SHAP分析結果/weather_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 上位10個の特徴量の詳細プロット
    mean_shap = np.abs(shap_values).mean(0)
    top_indices = np.argsort(mean_shap)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[:, top_indices], X_scaled[:, top_indices], 
                     feature_names=top_features, show=False)
    plt.title('SHAP Summary Plot - 上位10特徴量', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('SHAP分析結果/top10_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 個別サンプルの詳細プロット（最初の5サンプル）
    for i in range(min(5, len(X_scaled))):
        plt.figure(figsize=(12, 8))
        shap.force_plot(explainer.expected_value, shap_values[i], X_scaled[i], 
                       feature_names=feature_names, show=False)
        plt.title(f'SHAP Force Plot - サンプル {i+1}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'SHAP分析結果/force_plot_sample_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_weather_importance(shap_values, feature_names):
    """天気概況の重要度を詳細分析"""
    print("=== 天気概況の重要度分析 ===")
    
    # 天気概況関連の特徴量を特定
    weather_features = [f for f in feature_names if 'weather' in f.lower() or 'encoded' in f.lower()]
    
    if not weather_features:
        print("天気概況関連の特徴量が見つかりませんでした")
        return
    
    # 各天気概況特徴量の重要度を計算
    weather_importance = {}
    for feature in weather_features:
        idx = feature_names.index(feature)
        importance = np.abs(shap_values[:, idx]).mean()
        weather_importance[feature] = importance
    
    # 重要度でソート
    sorted_weather = sorted(weather_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== 天気概況特徴量の重要度 ===")
    for feature, importance in sorted_weather:
        print(f"{feature}: {importance:.6f}")
    
    # 可視化
    plt.figure(figsize=(10, 6))
    features = [item[0] for item in sorted_weather]
    importances = [item[1] for item in sorted_weather]
    
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel('SHAP重要度')
    plt.title('天気概況特徴量のSHAP重要度')
    plt.tight_layout()
    plt.savefig('SHAP分析結果/weather_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return weather_importance

def create_detailed_report(shap_values, feature_names, weather_importance):
    """詳細な分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    # 全体的な特徴量重要度
    mean_shap = np.abs(shap_values).mean(0)
    overall_importance = dict(zip(feature_names, mean_shap))
    sorted_overall = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True)
    
    # レポートファイルを作成
    with open('SHAP分析結果/詳細分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# SHAP分析結果レポート\n\n')
        
        f.write('## 概要\n')
        f.write(f'- 分析対象特徴量数: {len(feature_names)}\n')
        f.write(f'- 天気概況関連特徴量数: {len(weather_importance)}\n')
        f.write(f'- サンプル数: {len(shap_values)}\n\n')
        
        f.write('## 全体的な特徴量重要度（上位20個）\n')
        f.write('| 順位 | 特徴量名 | SHAP重要度 |\n')
        f.write('|------|----------|------------|\n')
        for i, (feature, importance) in enumerate(sorted_overall[:20], 1):
            f.write(f'| {i} | {feature} | {importance:.6f} |\n')
        
        f.write('\n## 天気概況関連特徴量の重要度\n')
        f.write('| 順位 | 特徴量名 | SHAP重要度 |\n')
        f.write('|------|----------|------------|\n')
        for i, (feature, importance) in enumerate(weather_importance.items(), 1):
            f.write(f'| {i} | {feature} | {importance:.6f} |\n')
        
        f.write('\n## 医学的解釈\n')
        f.write('- 天気概況の貢献度が低い場合、他の気象要素（温度、湿度、気圧）の方が重要\n')
        f.write('- 天気概況は解釈可能性向上のための追加機能として機能\n')
        f.write('- 動的最適化で除外されるのは正常な動作\n')
    
    print("詳細レポートを保存しました: SHAP分析結果/詳細分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        # 結果ディレクトリを作成
        import os
        os.makedirs('SHAP分析結果', exist_ok=True)
        
        # モデルとデータを読み込み
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        # データを準備
        print("データを準備中...")
        df, feature_columns, weather_encoder = prepare_data_for_shap(df)
        
        # SHAP分析を実行
        shap_values, X_scaled, y, feature_names, explainer = perform_shap_analysis(model, df, feature_columns)
        
        # SHAPプロットを作成
        create_shap_plots(shap_values, X_scaled, y, feature_names, explainer)
        
        # 天気概況の重要度を分析
        weather_importance = analyze_weather_importance(shap_values, feature_names)
        
        # 詳細レポートを作成
        create_detailed_report(shap_values, feature_names, weather_importance)
        
        print("\n=== SHAP分析完了 ===")
        print("結果は 'SHAP分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 