import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_ami_model_and_data():
    """AMIモデルとデータを読み込み"""
    print("=== AMIモデルとデータの読み込み ===")
    
    try:
        # 保存されたXGBoostモデルを読み込み
        model = joblib.load('saved_models/xgb_model_latest.pkl')
        print("✓ XGBoostモデル読み込み完了")
        
        # データの読み込み（元のスクリプトと同じ処理）
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people']
        
        print("✓ データ読み込み完了")
        return model, df
        
    except Exception as e:
        print(f"❌ モデルまたはデータの読み込みエラー: {e}")
        return None, None

def prepare_features_for_shap(df):
    """SHAP解析用の特徴量を準備"""
    print("\n=== SHAP解析用特徴量準備 ===")
    
    # 元のスクリプトと同じ特徴量作成処理
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
    
    # 異常気象フラグ（簡略版）
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    df['is_summer_day'] = (df['max_temp'] >= 25).astype(int)
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    
    # 気象相互作用
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['vapor_pressure']
    df['temp_change'] = (df['avg_temp'] - df['avg_temp'].shift(1)).fillna(0)
    df['humidity_change'] = (df['avg_humidity'] - df['avg_humidity'].shift(1)).fillna(0)
    df['pressure_change'] = (df['vapor_pressure'] - df['vapor_pressure'].shift(1)).fillna(0)
    
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    # 時系列特徴量
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    # ターゲット変数
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    print("✓ 特徴量作成完了")
    return df

def select_features_for_shap(df):
    """SHAP解析用の特徴量を選択"""
    print("\n=== 特徴量選択 ===")
    
    # 除外する列
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 
                   'hospitalization_count', 'people', 'hospitalization_date_af', 'people_af']
    
    # 特徴量列を取得
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
    
    print(f"✓ 選択された特徴量数: {len(feature_columns)}")
    return feature_columns

def perform_shap_analysis(model, df, feature_columns):
    """SHAP解析を実行"""
    print("\n=== SHAP解析実行 ===")
    
    # データの準備
    X = df[feature_columns].copy()
    y = df['target']
    
    # NaN値の処理
    X = X.fillna(method='ffill').fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    print(f"✓ データ準備完了: {X_scaled.shape}")
    
    # SHAP値の計算
    print("SHAP値を計算中...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    print("✓ SHAP値計算完了")
    
    return X_scaled, y, shap_values, explainer

def create_shap_visualizations(X_scaled, y, shap_values, explainer, feature_columns):
    """SHAP可視化を作成"""
    print("\n=== SHAP可視化作成 ===")
    
    # 結果保存用ディレクトリ作成
    import os
    os.makedirs('shap_results', exist_ok=True)
    
    # 1. Summary Plot
    print("1. Summary Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_columns, show=False)
    plt.title('SHAP Summary Plot - AMI Weather Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Summary Plot保存完了")
    
    # 2. Feature Importance Plot
    print("2. Feature Importance Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_scaled, plot_type="bar", feature_names=feature_columns, show=False)
    plt.title('SHAP Feature Importance - AMI Weather Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_results/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature Importance Plot保存完了")
    
    # 3. Dependence Plots for Top Features
    print("3. Dependence Plots作成中...")
    # 上位10個の特徴量を取得
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-10:]
    top_features = [feature_columns[i] for i in top_features_idx]
    
    for i, feature in enumerate(top_features):
        print(f"  {i+1}/10: {feature}")
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_scaled, feature_names=feature_columns, show=False)
        plt.title(f'SHAP Dependence Plot - {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_results/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Dependence Plots保存完了")
    
    # 4. Waterfall Plot for Sample Cases
    print("4. Waterfall Plots作成中...")
    # 高リスク日のサンプル
    high_risk_indices = y[y == 1].index[:5]
    for i, idx in enumerate(high_risk_indices):
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explainer.expected_value, shap_values[idx], X_scaled.iloc[idx], show=False)
        plt.title(f'SHAP Waterfall Plot - High Risk Case {i+1}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_results/shap_waterfall_high_risk_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 低リスク日のサンプル
    low_risk_indices = y[y == 0].index[:5]
    for i, idx in enumerate(low_risk_indices):
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explainer.expected_value, shap_values[idx], X_scaled.iloc[idx], show=False)
        plt.title(f'SHAP Waterfall Plot - Low Risk Case {i+1}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_results/shap_waterfall_low_risk_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Waterfall Plots保存完了")

def create_shap_report(X_scaled, y, shap_values, feature_columns):
    """SHAP解析レポートを作成"""
    print("\n=== SHAP解析レポート作成 ===")
    
    # 特徴量重要度の計算
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 上位20個の特徴量
    top_20_features = importance_df.head(20)
    
    # レポート作成
    report = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'XGBoost',
        'target_variable': 'AMI High Risk Days (75th percentile)',
        'data_shape': X_scaled.shape,
        'positive_class_ratio': y.mean(),
        'top_20_features': top_20_features.to_dict('records'),
        'feature_importance_summary': {
            'total_features': len(feature_columns),
            'weather_features': len([f for f in feature_columns if any(x in f for x in ['temp', 'humidity', 'pressure', 'wind', 'sunshine'])]),
            'extreme_weather_features': len([f for f in feature_columns if 'is_' in f]),
            'time_series_features': len([f for f in feature_columns if any(x in f for x in ['ma_', 'std_', 'lag_'])]),
            'interaction_features': len([f for f in feature_columns if any(x in f for x in ['temp_', 'humidity_', 'pressure_'])]),
        }
    }
    
    # JSONファイルとして保存
    import json
    with open('shap_results/shap_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # Markdownファイルとして保存
    with open('shap_results/shap_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('# AMI Weather Model SHAP Analysis Report\n\n')
        f.write(f'**Analysis Date:** {report["analysis_date"]}\n\n')
        f.write(f'**Model Type:** {report["model_type"]}\n\n')
        f.write(f'**Target Variable:** {report["target_variable"]}\n\n')
        f.write(f'**Data Shape:** {report["data_shape"]}\n\n')
        f.write(f'**Positive Class Ratio:** {report["positive_class_ratio"]:.3f}\n\n')
        
        f.write('## Feature Importance Summary\n\n')
        f.write('| Category | Count |\n')
        f.write('|----------|-------|\n')
        for category, count in report['feature_importance_summary'].items():
            f.write(f'| {category.replace("_", " ").title()} | {count} |\n')
        
        f.write('\n## Top 20 Most Important Features\n\n')
        f.write('| Rank | Feature | Importance |\n')
        f.write('|------|---------|------------|\n')
        for i, row in enumerate(top_20_features.itertuples(), 1):
            f.write(f'| {i} | {row.feature} | {row.importance:.4f} |\n')
    
    print("✓ SHAP解析レポート保存完了")
    return report

def main():
    """メイン実行関数"""
    print("=== AMI Weather Model SHAP Analysis ===")
    
    # 1. モデルとデータの読み込み
    model, df = load_ami_model_and_data()
    if model is None or df is None:
        print("❌ モデルまたはデータの読み込みに失敗しました")
        return
    
    # 2. 特徴量の準備
    df = prepare_features_for_shap(df)
    feature_columns = select_features_for_shap(df)
    
    # 3. SHAP解析の実行
    X_scaled, y, shap_values, explainer = perform_shap_analysis(model, df, feature_columns)
    
    # 4. 可視化の作成
    create_shap_visualizations(X_scaled, y, shap_values, explainer, feature_columns)
    
    # 5. レポートの作成
    report = create_shap_report(X_scaled, y, shap_values, feature_columns)
    
    print("\n=== SHAP解析完了 ===")
    print("✓ 結果は 'shap_results/' ディレクトリに保存されました")
    print(f"✓ 上位特徴量: {report['top_20_features'][0]['feature']}")

if __name__ == "__main__":
    main() 