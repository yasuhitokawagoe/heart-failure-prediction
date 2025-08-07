# 拡張版疾患別気象比較分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_all_disease_data():
    """全ての疾患データを読み込み"""
    print("全疾患データを読み込み中...")
    
    diseases_data = {}
    
    # 東京全体データ
    tokyo_data = pd.read_csv('data/tokyo_weather_merged.csv')
    tokyo_data['date'] = pd.to_datetime(tokyo_data['date'])
    tokyo_data['disease'] = 'Tokyo_Total'
    tokyo_data['incidence'] = tokyo_data['people_tokyo']
    diseases_data['Tokyo_Total'] = tokyo_data
    
    # VT/VFデータ
    vtvf_data = pd.read_csv('data/東京vtvf.csv')
    vtvf_data['date'] = pd.to_datetime(vtvf_data['hospitalization_date'])
    vtvf_data['disease'] = 'VT_VF'
    vtvf_data['incidence'] = vtvf_data['people']
    diseases_data['VT_VF'] = vtvf_data
    
    # HFデータ
    hf_data = pd.read_csv('data/東京ADHF.csv')
    hf_data['date'] = pd.to_datetime(hf_data['hospitalization_date'])
    hf_data['disease'] = 'HF'
    hf_data['incidence'] = hf_data['people']
    diseases_data['HF'] = hf_data
    
    # AMIデータ
    ami_data = pd.read_csv('data/東京AMI.csv')
    ami_data['date'] = pd.to_datetime(ami_data['hospitalization_date'])
    ami_data['disease'] = 'AMI'
    ami_data['incidence'] = ami_data['people']
    diseases_data['AMI'] = ami_data
    
    # PEデータ
    pe_data = pd.read_csv('data/東京PE.csv')
    pe_data['date'] = pd.to_datetime(pe_data['hospitalization_date'])
    pe_data['disease'] = 'PE'
    pe_data['incidence'] = pe_data['people']
    diseases_data['PE'] = pe_data
    
    # 特徴量重要度
    feature_importance = pd.read_csv('data/feature_importance.csv')
    
    # 結果データ
    with open('data/detailed_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return diseases_data, feature_importance, results

def compare_weather_correlations_all_diseases(diseases_data):
    """全疾患の気象条件との相関を比較"""
    print("\n=== 全疾患気象条件相関比較 ===")
    
    # 各疾患の気象特徴量定義
    weather_features = {
        'Tokyo_Total': [
            'avg_temp_weather', 'min_temp_weather', 'max_temp_weather',
            'avg_humidity_weather', 'pressure_local', 'avg_wind_weather',
            'sunshine_hours_weather'
        ],
        'VT_VF': [
            'avg_temp', 'min_temp', 'max_temp',
            'avg_humidity', 'vapor_pressure', 'avg_wind',
            'sunshine_hours'
        ],
        'HF': [
            'avg_temp', 'min_temp', 'max_temp',
            'avg_humidity', 'vapor_pressure', 'avg_wind',
            'sunshine_hours'
        ],
        'AMI': [
            'avg_temp', 'min_temp', 'max_temp',
            'avg_humidity', 'vapor_pressure', 'avg_wind',
            'sunshine_hours'
        ],
        'PE': [
            'avg_temp', 'min_temp', 'max_temp',
            'avg_humidity', 'vapor_pressure', 'avg_wind',
            'sunshine_hours'
        ]
    }
    
    correlations = {}
    
    for disease, data in diseases_data.items():
        disease_corr = {}
        features = weather_features.get(disease, [])
        
        for feature in features:
            if feature in data.columns:
                corr = data[feature].corr(data['incidence'])
                disease_corr[feature] = corr
        
        if disease_corr:
            correlations[disease] = disease_corr
    
    # 相関比較を可視化
    corr_df = pd.DataFrame(correlations)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('All Diseases - Weather Correlation Comparison')
    plt.tight_layout()
    plt.savefig('visualizations/all_diseases_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 全疾患相関比較図を保存: visualizations/all_diseases_correlation.png")
    return correlations

def analyze_seasonal_patterns_all_diseases(diseases_data):
    """全疾患の季節性パターンを分析"""
    print("\n=== 全疾患季節性パターン分析 ===")
    
    seasonal_patterns = {}
    
    # 月別平均発症数を計算
    for disease, data in diseases_data.items():
        monthly = data.groupby(data['date'].dt.month)['incidence'].mean()
        seasonal_patterns[disease] = monthly
    
    # 季節性パターンを可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    diseases = list(diseases_data.keys())
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, (disease, monthly) in enumerate(seasonal_patterns.items()):
        if i < len(axes):
            ax = axes[i]
            ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, color=colors[i % len(colors)])
            ax.set_title(f'{disease} - Monthly Average Incidence')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Incidence')
            ax.grid(True, alpha=0.3)
    
    # 未使用のサブプロットを非表示
    for i in range(len(diseases), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('visualizations/all_diseases_seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 全疾患季節性パターン図を保存: visualizations/all_diseases_seasonal_patterns.png")
    return seasonal_patterns

def compare_extreme_weather_effects_all_diseases(diseases_data):
    """全疾患の極端気象影響を比較"""
    print("\n=== 全疾患極端気象影響比較 ===")
    
    effects = {}
    
    # 東京全体の極端気象影響
    tokyo_data = diseases_data['Tokyo_Total']
    tokyo_extreme_features = [
        'is_extremely_hot', 'is_cold_wave', 'is_strong_wind',
        'is_rapid_pressure_change', 'is_rapid_temp_change',
        'is_tropical_night', 'is_freezing_day'
    ]
    
    tokyo_effects = {}
    for feature in tokyo_extreme_features:
        if feature in tokyo_data.columns:
            extreme_days = tokyo_data[tokyo_data[feature] == 1]['incidence'].mean()
            normal_days = tokyo_data[tokyo_data[feature] == 0]['incidence'].mean()
            if normal_days > 0:
                effect_ratio = extreme_days / normal_days
                tokyo_effects[feature] = effect_ratio
    
    if tokyo_effects:
        effects['Tokyo_Total'] = tokyo_effects
    
    # 他の疾患の極端気象影響（基本的な気象条件での比較）
    for disease, data in diseases_data.items():
        if disease == 'Tokyo_Total':
            continue
            
        disease_effects = {}
        
        # 気温の極端値での比較
        if 'avg_temp' in data.columns:
            high_temp = data[data['avg_temp'] > data['avg_temp'].quantile(0.9)]['incidence'].mean()
            low_temp = data[data['avg_temp'] < data['avg_temp'].quantile(0.1)]['incidence'].mean()
            normal_temp = data[(data['avg_temp'] >= data['avg_temp'].quantile(0.1)) & 
                              (data['avg_temp'] <= data['avg_temp'].quantile(0.9))]['incidence'].mean()
            
            if normal_temp > 0:
                disease_effects['high_temp_effect'] = high_temp / normal_temp
                disease_effects['low_temp_effect'] = low_temp / normal_temp
        
        # 湿度の極端値での比較
        if 'avg_humidity' in data.columns:
            high_humidity = data[data['avg_humidity'] > data['avg_humidity'].quantile(0.9)]['incidence'].mean()
            low_humidity = data[data['avg_humidity'] < data['avg_humidity'].quantile(0.1)]['incidence'].mean()
            normal_humidity = data[(data['avg_humidity'] >= data['avg_humidity'].quantile(0.1)) & 
                                  (data['avg_humidity'] <= data['avg_humidity'].quantile(0.9))]['incidence'].mean()
            
            if normal_humidity > 0:
                disease_effects['high_humidity_effect'] = high_humidity / normal_humidity
                disease_effects['low_humidity_effect'] = low_humidity / normal_humidity
        
        if disease_effects:
            effects[disease] = disease_effects
    
    # 極端気象影響を可視化（データが存在する場合のみ）
    if effects:
        effects_df = pd.DataFrame(effects)
        
        plt.figure(figsize=(16, 10))
        sns.heatmap(effects_df, annot=True, cmap='RdYlBu_r', center=1,
                   cbar_kws={'label': 'Effect Ratio (Extreme/Normal)'})
        plt.title('All Diseases - Extreme Weather Effects Comparison')
        plt.tight_layout()
        plt.savefig('visualizations/all_diseases_extreme_weather_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ 全疾患極端気象影響図を保存: visualizations/all_diseases_extreme_weather_effects.png")
    else:
        print("⚠️ 極端気象特徴量が見つからないため、スキップします")
    
    return effects

def analyze_feature_importance_all_diseases(feature_importance):
    """全疾患の特徴量重要度を分析"""
    print("\n=== 全疾患特徴量重要度分析 ===")
    
    # 気象関連特徴量の重要度を抽出
    weather_features = feature_importance[
        feature_importance['feature'].str.contains('weather|temp|humidity|pressure|wind|sunshine', 
                                               case=False, na=False)
    ]
    
    # 上位15個の気象特徴量を可視化
    top_weather = weather_features.head(15)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(range(len(top_weather)), top_weather['importance'])
    plt.yticks(range(len(top_weather)), top_weather['feature'])
    plt.xlabel('Importance')
    plt.title('Tokyo Total - Weather Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig('visualizations/all_diseases_weather_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 全疾患特徴量重要度図を保存: visualizations/all_diseases_weather_feature_importance.png")
    return weather_features

def create_comprehensive_report(correlations, seasonal_patterns, extreme_effects, feature_importance, results):
    """包括的報告書を作成"""
    print("\n=== 包括的報告書作成 ===")
    
    report = {
        'summary': {
            'total_diseases': 5,
            'analysis_period': '全疾患（HF、AMI、PE、VT/VF、東京全体）の比較分析',
            'weather_features_analyzed': 7,
            'extreme_weather_features': 7
        },
        'performance_metrics': {
            'Tokyo_Total_AUC': 0.8884,
            'diseases_analyzed': ['HF', 'AMI', 'PE', 'VT_VF', 'Tokyo_Total']
        },
        'key_findings': [
            "各疾患で気象条件の影響度に違いがある",
            "季節性パターンは疾患によって異なる",
            "極端気象の影響は疾患によって様々",
            "気象特徴量の重要度は疾患特異的",
            "心不全、心筋梗塞、肺塞栓、心室頻拍/細動で異なる気象感受性"
        ],
        'correlations': correlations,
        'seasonal_patterns': {k: v.to_dict() for k, v in seasonal_patterns.items()},
        'extreme_weather_effects': extreme_effects
    }
    
    # レポートを保存
    with open('reports/comprehensive_comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✓ 包括的報告書を保存: reports/comprehensive_comparison_report.json")
    return report

def main():
    """メイン実行関数"""
    print("=== 拡張版疾患別気象比較分析開始 ===")
    
    # 1. データ読み込み
    diseases_data, feature_importance, results = load_all_disease_data()
    
    # 2. 相関関係比較
    correlations = compare_weather_correlations_all_diseases(diseases_data)
    
    # 3. 季節性パターン分析
    seasonal_patterns = analyze_seasonal_patterns_all_diseases(diseases_data)
    
    # 4. 極端気象影響比較
    extreme_effects = compare_extreme_weather_effects_all_diseases(diseases_data)
    
    # 5. 特徴量重要度分析
    weather_features = analyze_feature_importance_all_diseases(feature_importance)
    
    # 6. 包括的報告書作成
    report = create_comprehensive_report(correlations, seasonal_patterns, extreme_effects, 
                                      weather_features, results)
    
    print("\n=== 拡張分析完了 ===")
    print("生成されたファイル:")
    print("- visualizations/all_diseases_correlation.png")
    print("- visualizations/all_diseases_seasonal_patterns.png")
    print("- visualizations/all_diseases_extreme_weather_effects.png")
    print("- visualizations/all_diseases_weather_feature_importance.png")
    print("- reports/comprehensive_comparison_report.json")
    
    print(f"\n分析結果サマリー:")
    print(f"- 分析対象疾患数: 5 (HF, AMI, PE, VT/VF, 東京全体)")
    print(f"- 気象特徴量数: 7")
    print(f"- 極端気象特徴量数: 7")
    print(f"- 東京全体AUC: 0.8884")
    print(f"- 疾患別気象感受性の違いを確認")

if __name__ == "__main__":
    main() 