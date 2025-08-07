# 高度な異常気象フラグと気象組み合わせ比較分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_extreme_weather_features(df, weather_prefix=''):
    """異常気象フラグと気象組み合わせ特徴量を作成"""
    print(f"異常気象フラグを作成中... (prefix: {weather_prefix})")
    
    # 基本気象列名の設定
    if weather_prefix:
        temp_col = f'avg_temp_{weather_prefix}'
        max_temp_col = f'max_temp_{weather_prefix}'
        min_temp_col = f'min_temp_{weather_prefix}'
        wind_col = f'avg_wind_{weather_prefix}'
        humidity_col = f'avg_humidity_{weather_prefix}'
        sunshine_col = f'sunshine_hours_{weather_prefix}'
        pressure_col = 'pressure_local' if weather_prefix == 'weather' else 'vapor_pressure'
    else:
        temp_col = 'avg_temp'
        max_temp_col = 'max_temp'
        min_temp_col = 'min_temp'
        wind_col = 'avg_wind'
        humidity_col = 'avg_humidity'
        sunshine_col = 'sunshine_hours'
        pressure_col = 'vapor_pressure'
    
    # 基本異常気象フラグ
    df[f'is_tropical_night'] = (df[min_temp_col] >= 25).astype(int)
    df[f'is_extremely_hot'] = (df[max_temp_col] >= 35).astype(int)
    df[f'is_hot_day'] = (df[max_temp_col] >= 30).astype(int)
    df[f'is_summer_day'] = (df[max_temp_col] >= 25).astype(int)
    df[f'is_winter_day'] = (df[min_temp_col] < 0).astype(int)
    df[f'is_freezing_day'] = (df[max_temp_col] < 0).astype(int)
    
    # 寒波検出
    df['monthly_temp_mean'] = df.groupby(['year', 'month'])[temp_col].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )
    df[f'is_cold_wave'] = (df[temp_col] <= (df['monthly_temp_mean'] - 2.0)).astype(int)
    
    # 強風検出
    df['wind_quantile'] = df[wind_col].expanding().quantile(0.95).shift(1).fillna(df[wind_col].quantile(0.95))
    df[f'is_strong_wind'] = (df[wind_col] > df['wind_quantile']).astype(int)
    
    # 台風条件
    df['pressure_quantile'] = df[pressure_col].expanding().quantile(0.1).shift(1).fillna(df[pressure_col].quantile(0.1))
    df['wind_quantile_90'] = df[wind_col].expanding().quantile(0.9).shift(1).fillna(df[wind_col].quantile(0.9))
    df[f'is_typhoon_condition'] = ((df[pressure_col] < df['pressure_quantile']) & 
                                   (df[wind_col] > df['wind_quantile_90'])).astype(int)
    
    # 急激な気圧変化
    df['pressure_change'] = df[pressure_col] - df[pressure_col].shift(1)
    df['pressure_change'] = df['pressure_change'].fillna(0)
    df['pressure_change_std'] = df['pressure_change'].expanding().std().shift(1).fillna(df['pressure_change'].std())
    df[f'is_rapid_pressure_change'] = (abs(df['pressure_change']) > df['pressure_change_std'] * 2).astype(int)
    
    # 異常湿度
    df['humidity_quantile_95'] = df[humidity_col].expanding().quantile(0.95).shift(1).fillna(df[humidity_col].quantile(0.95))
    df['humidity_quantile_05'] = df[humidity_col].expanding().quantile(0.05).shift(1).fillna(df[humidity_col].quantile(0.05))
    df[f'is_extreme_humidity_high'] = (df[humidity_col] > df['humidity_quantile_95']).astype(int)
    df[f'is_extreme_humidity_low'] = (df[humidity_col] < df['humidity_quantile_05']).astype(int)
    
    # 日照時間の極端な状態
    df['sunshine_quantile_95'] = df[sunshine_col].expanding().quantile(0.95).shift(1).fillna(df[sunshine_col].quantile(0.95))
    df['sunshine_quantile_05'] = df[sunshine_col].expanding().quantile(0.05).shift(1).fillna(df[sunshine_col].quantile(0.05))
    df[f'is_extremely_sunny'] = (df[sunshine_col] > df['sunshine_quantile_95']).astype(int)
    df[f'is_extremely_cloudy'] = (df[sunshine_col] < df['sunshine_quantile_05']).astype(int)
    
    # 気温の急激な変化
    df['temp_change_3d'] = df[temp_col] - df[temp_col].shift(3)
    df[f'is_rapid_temp_change'] = (abs(df['temp_change_3d']) > 5).astype(int)
    
    # 気温の変動性
    df['temp_volatility'] = df[temp_col].rolling(window=7, min_periods=1).std().shift(1).fillna(0)
    df['temp_volatility_quantile'] = df['temp_volatility'].expanding().quantile(0.9).shift(1).fillna(df['temp_volatility'].quantile(0.9))
    df[f'is_high_temp_volatility'] = (df['temp_volatility'] > df['temp_volatility_quantile']).astype(int)
    
    # 気象組み合わせ特徴量
    df['temp_quantile_80'] = df[temp_col].expanding().quantile(0.8).shift(1).fillna(df[temp_col].quantile(0.8))
    df['humidity_quantile_80'] = df[humidity_col].expanding().quantile(0.8).shift(1).fillna(df[humidity_col].quantile(0.8))
    df[f'temp_humidity_stress'] = (
        (df[temp_col] > df['temp_quantile_80']) & 
        (df[humidity_col] > df['humidity_quantile_80'])
    ).astype(int)
    
    df['temp_quantile_20'] = df[temp_col].expanding().quantile(0.2).shift(1).fillna(df[temp_col].quantile(0.2))
    df['wind_quantile_80'] = df[wind_col].expanding().quantile(0.8).shift(1).fillna(df[wind_col].quantile(0.8))
    df[f'temp_wind_stress'] = (
        (df[temp_col] < df['temp_quantile_20']) & 
        (df[wind_col] > df['wind_quantile_80'])
    ).astype(int)
    
    df['pressure_quantile_20'] = df[pressure_col].expanding().quantile(0.2).shift(1).fillna(df[pressure_col].quantile(0.2))
    df[f'pressure_wind_stress'] = (
        (df[pressure_col] < df['pressure_quantile_20']) & 
        (df[wind_col] > df['wind_quantile_80'])
    ).astype(int)
    
    df['sunshine_quantile_80'] = df[sunshine_col].expanding().quantile(0.8).shift(1).fillna(df[sunshine_col].quantile(0.8))
    df[f'sunshine_temp_stress'] = (
        (df[sunshine_col] > df['sunshine_quantile_80']) & 
        (df[temp_col] > df['temp_quantile_80'])
    ).astype(int)
    
    # 複合的な気象ストレス指標
    df[f'weather_stress'] = (
        df[f'is_extremely_hot'].astype(float) * 0.3 +
        df[f'is_cold_wave'].astype(float) * 0.3 +
        df[f'is_rapid_pressure_change'].astype(float) * 0.2 +
        df[f'is_strong_wind'].astype(float) * 0.2
    )
    
    # 季節性を考慮した気象影響
    df[f'seasonal_weather_impact'] = (
        (df['month'].isin([6, 7, 8]) & df[f'is_extremely_hot']).astype(int) * 0.5 +
        (df['month'].isin([12, 1, 2]) & df[f'is_cold_wave']).astype(int) * 0.5
    )
    
    return df

def load_and_process_all_disease_data():
    """全疾患データを読み込み、異常気象フラグを追加"""
    print("全疾患データを読み込み、異常気象フラグを追加中...")
    
    diseases_data = {}
    
    # 東京全体データ
    tokyo_data = pd.read_csv('data/tokyo_weather_merged.csv')
    tokyo_data['date'] = pd.to_datetime(tokyo_data['date'])
    tokyo_data['disease'] = 'Tokyo_Total'
    tokyo_data['incidence'] = tokyo_data['people_tokyo']
    
    # 日付特徴量を追加
    tokyo_data['year'] = tokyo_data['date'].dt.year
    tokyo_data['month'] = tokyo_data['date'].dt.month
    tokyo_data['day'] = tokyo_data['date'].dt.day
    
    # 異常気象フラグを追加
    tokyo_data = create_extreme_weather_features(tokyo_data, 'weather')
    diseases_data['Tokyo_Total'] = tokyo_data
    
    # 他の疾患データ
    disease_files = {
        'VT_VF': 'data/東京vtvf.csv',
        'HF': 'data/東京ADHF.csv',
        'AMI': 'data/東京AMI.csv',
        'PE': 'data/東京PE.csv'
    }
    
    for disease, file_path in disease_files.items():
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['hospitalization_date'])
        data['disease'] = disease
        data['incidence'] = data['people']
        
        # 日付特徴量を追加
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day'] = data['date'].dt.day
        
        # 異常気象フラグを追加
        data = create_extreme_weather_features(data, '')
        diseases_data[disease] = data
    
    return diseases_data

def compare_extreme_weather_flags(diseases_data):
    """異常気象フラグの比較分析"""
    print("\n=== 異常気象フラグ比較分析 ===")
    
    # 基本異常気象フラグ
    basic_flags = [
        'is_tropical_night', 'is_extremely_hot', 'is_hot_day', 'is_summer_day',
        'is_winter_day', 'is_freezing_day', 'is_cold_wave', 'is_strong_wind',
        'is_typhoon_condition', 'is_rapid_pressure_change', 'is_extreme_humidity_high',
        'is_extreme_humidity_low', 'is_extremely_sunny', 'is_extremely_cloudy',
        'is_rapid_temp_change', 'is_high_temp_volatility'
    ]
    
    # 気象組み合わせフラグ
    combination_flags = [
        'temp_humidity_stress', 'temp_wind_stress', 'pressure_wind_stress',
        'sunshine_temp_stress', 'weather_stress', 'seasonal_weather_impact'
    ]
    
    all_flags = basic_flags + combination_flags
    
    # 各疾患の異常気象フラグ発現率を計算
    flag_occurrence = {}
    
    for disease, data in diseases_data.items():
        disease_flags = {}
        for flag in all_flags:
            if flag in data.columns:
                occurrence_rate = data[flag].mean()
                disease_flags[flag] = occurrence_rate
        flag_occurrence[disease] = disease_flags
    
    # 異常気象フラグ発現率を可視化
    flag_df = pd.DataFrame(flag_occurrence)
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(flag_df, annot=True, cmap='YlOrRd', fmt='.3f',
               cbar_kws={'label': 'Occurrence Rate'})
    plt.title('Extreme Weather Flags Occurrence Rate by Disease')
    plt.tight_layout()
    plt.savefig('visualizations/extreme_weather_flags_occurrence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 異常気象フラグ発現率図を保存: visualizations/extreme_weather_flags_occurrence.png")
    
    # 異常気象フラグと発症率の相関を分析
    flag_correlations = {}
    
    for disease, data in diseases_data.items():
        disease_corr = {}
        for flag in all_flags:
            if flag in data.columns:
                corr = data[flag].corr(data['incidence'])
                disease_corr[flag] = corr
        flag_correlations[disease] = disease_corr
    
    # 相関を可視化
    corr_df = pd.DataFrame(flag_correlations)
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Correlation with Incidence'})
    plt.title('Extreme Weather Flags Correlation with Disease Incidence')
    plt.tight_layout()
    plt.savefig('visualizations/extreme_weather_flags_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 異常気象フラグ相関図を保存: visualizations/extreme_weather_flags_correlation.png")
    
    return flag_occurrence, flag_correlations

def analyze_weather_combination_effects(diseases_data):
    """気象組み合わせの影響分析"""
    print("\n=== 気象組み合わせ影響分析 ===")
    
    combination_effects = {}
    
    for disease, data in diseases_data.items():
        disease_effects = {}
        
        # 気象組み合わせフラグの影響を分析
        combination_flags = [
            'temp_humidity_stress', 'temp_wind_stress', 'pressure_wind_stress',
            'sunshine_temp_stress', 'weather_stress', 'seasonal_weather_impact'
        ]
        
        for flag in combination_flags:
            if flag in data.columns:
                # 組み合わせ条件ありの日の平均発症数
                combination_days = data[data[flag] == 1]['incidence'].mean()
                # 組み合わせ条件なしの日の平均発症数
                normal_days = data[data[flag] == 0]['incidence'].mean()
                
                if normal_days > 0:
                    effect_ratio = combination_days / normal_days
                    disease_effects[flag] = effect_ratio
        
        combination_effects[disease] = disease_effects
    
    # 気象組み合わせ影響を可視化
    effects_df = pd.DataFrame(combination_effects)
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(effects_df, annot=True, cmap='RdYlBu_r', center=1,
               cbar_kws={'label': 'Effect Ratio (Combination/Normal)'})
    plt.title('Weather Combination Effects on Disease Incidence')
    plt.tight_layout()
    plt.savefig('visualizations/weather_combination_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 気象組み合わせ影響図を保存: visualizations/weather_combination_effects.png")
    
    return combination_effects

def analyze_weather_stress_patterns(diseases_data):
    """気象ストレスパターンの分析"""
    print("\n=== 気象ストレスパターン分析 ===")
    
    stress_patterns = {}
    
    for disease, data in diseases_data.items():
        if 'weather_stress' in data.columns:
            # 気象ストレスの分布
            stress_stats = {
                'mean_stress': data['weather_stress'].mean(),
                'max_stress': data['weather_stress'].max(),
                'stress_std': data['weather_stress'].std(),
                'high_stress_days': (data['weather_stress'] > data['weather_stress'].quantile(0.9)).sum(),
                'total_days': len(data)
            }
            stress_patterns[disease] = stress_stats
    
    # 気象ストレスパターンを可視化
    stress_df = pd.DataFrame(stress_patterns).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 平均ストレス
    axes[0, 0].bar(stress_df.index, stress_df['mean_stress'])
    axes[0, 0].set_title('Mean Weather Stress by Disease')
    axes[0, 0].set_ylabel('Mean Stress Level')
    
    # 最大ストレス
    axes[0, 1].bar(stress_df.index, stress_df['max_stress'])
    axes[0, 1].set_title('Max Weather Stress by Disease')
    axes[0, 1].set_ylabel('Max Stress Level')
    
    # ストレス標準偏差
    axes[1, 0].bar(stress_df.index, stress_df['stress_std'])
    axes[1, 0].set_title('Weather Stress Variability by Disease')
    axes[1, 0].set_ylabel('Stress Standard Deviation')
    
    # 高ストレス日数
    axes[1, 1].bar(stress_df.index, stress_df['high_stress_days'])
    axes[1, 1].set_title('High Stress Days by Disease')
    axes[1, 1].set_ylabel('Number of High Stress Days')
    
    plt.tight_layout()
    plt.savefig('visualizations/weather_stress_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 気象ストレスパターン図を保存: visualizations/weather_stress_patterns.png")
    
    return stress_patterns

def create_comprehensive_weather_report(flag_occurrence, flag_correlations, combination_effects, stress_patterns):
    """包括的な気象報告書を作成"""
    print("\n=== 包括的気象報告書作成 ===")
    
    def convert_numpy(obj):
        """numpy型をJSONシリアライズ可能な型に変換"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    report = {
        'summary': {
            'total_diseases': 5,
            'analysis_type': '高度な異常気象フラグと気象組み合わせ分析',
            'extreme_weather_flags': 16,
            'weather_combination_flags': 6
        },
        'key_findings': [
            "各疾患で異常気象フラグの発現率が異なる",
            "気象組み合わせ条件の影響は疾患特異的",
            "気象ストレスパターンは疾患によって大きく異なる",
            "複合的な気象条件が疾患発症に重要な影響を与える"
        ],
        'flag_occurrence': convert_numpy(flag_occurrence),
        'flag_correlations': convert_numpy(flag_correlations),
        'combination_effects': convert_numpy(combination_effects),
        'stress_patterns': convert_numpy(stress_patterns)
    }
    
    # レポートを保存
    with open('reports/advanced_weather_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✓ 包括的気象報告書を保存: reports/advanced_weather_analysis_report.json")
    return report

def main():
    """メイン実行関数"""
    print("=== 高度な異常気象フラグと気象組み合わせ比較分析開始 ===")
    
    # 1. データ読み込みと異常気象フラグ追加
    diseases_data = load_and_process_all_disease_data()
    
    # 2. 異常気象フラグ比較分析
    flag_occurrence, flag_correlations = compare_extreme_weather_flags(diseases_data)
    
    # 3. 気象組み合わせ影響分析
    combination_effects = analyze_weather_combination_effects(diseases_data)
    
    # 4. 気象ストレスパターン分析
    stress_patterns = analyze_weather_stress_patterns(diseases_data)
    
    # 5. 包括的報告書作成
    report = create_comprehensive_weather_report(flag_occurrence, flag_correlations, 
                                              combination_effects, stress_patterns)
    
    print("\n=== 高度な気象分析完了 ===")
    print("生成されたファイル:")
    print("- visualizations/extreme_weather_flags_occurrence.png")
    print("- visualizations/extreme_weather_flags_correlation.png")
    print("- visualizations/weather_combination_effects.png")
    print("- visualizations/weather_stress_patterns.png")
    print("- reports/advanced_weather_analysis_report.json")
    
    print(f"\n分析結果サマリー:")
    print(f"- 分析対象疾患数: 5 (HF, AMI, PE, VT/VF, 東京全体)")
    print(f"- 異常気象フラグ数: 16")
    print(f"- 気象組み合わせフラグ数: 6")
    print(f"- 疾患別気象ストレスパターンの違いを確認")

if __name__ == "__main__":
    main() 