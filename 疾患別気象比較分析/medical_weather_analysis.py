# 気象ストレスの医学的定義と臨床的意義分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def define_medical_weather_stress(df, weather_prefix=''):
    """医学的に定義された気象ストレス指標を作成"""
    print(f"医学的気象ストレス指標を作成中... (prefix: {weather_prefix})")
    
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
    
    # 1. 心血管系ストレス指標（医学的根拠に基づく）
    
    # 寒冷ストレス（血管収縮、血圧上昇）
    df['cold_stress'] = np.where(
        df[temp_col] < 10, 
        (10 - df[temp_col]) * 0.5,  # 10℃以下で線形増加
        0
    )
    
    # 暑熱ストレス（脱水、心拍数増加）
    df['heat_stress'] = np.where(
        df[max_temp_col] > 30,
        (df[max_temp_col] - 30) * 0.3,  # 30℃以上で線形増加
        0
    )
    
    # 湿度ストレス（発汗困難、体温調節障害）
    df['humidity_stress'] = np.where(
        df[humidity_col] > 70,
        (df[humidity_col] - 70) * 0.02,  # 70%以上で線形増加
        0
    )
    
    # 気圧変動ストレス（自律神経系への影響）
    df['pressure_change'] = df[pressure_col] - df[pressure_col].shift(1)
    df['pressure_change'] = df['pressure_change'].fillna(0)
    df['pressure_stress'] = np.abs(df['pressure_change']) * 0.1
    
    # 2. 疾患特異的ストレス指標
    
    # 心不全特異的ストレス
    df['hf_cold_stress'] = np.where(
        df[temp_col] < 15,  # 心不全患者は15℃以下で悪化
        (15 - df[temp_col]) * 0.8,
        0
    )
    
    df['hf_humidity_stress'] = np.where(
        df[humidity_col] > 60,  # 心不全患者は高湿度で悪化
        (df[humidity_col] - 60) * 0.03,
        0
    )
    
    # 心筋梗塞特異的ストレス
    df['ami_cold_stress'] = np.where(
        df[temp_col] < 12,  # 心筋梗塞は12℃以下でリスク増加
        (12 - df[temp_col]) * 1.0,
        0
    )
    
    df['ami_pressure_stress'] = np.where(
        abs(df['pressure_change']) > 5,  # 急激な気圧変化
        abs(df['pressure_change']) * 0.2,
        0
    )
    
    # VT/VF特異的ストレス
    df['vtvf_humidity_stress'] = np.where(
        df[humidity_col] > 75,  # VT/VFは高湿度でリスク増加
        (df[humidity_col] - 75) * 0.04,
        0
    )
    
    df['vtvf_temp_volatility'] = df[temp_col].rolling(window=3).std() * 0.5
    
    # 3. 複合ストレス指標（臨床的に意味のある組み合わせ）
    
    # 心血管複合ストレス（寒冷+高湿度）
    df['cardiovascular_composite_stress'] = (
        df['cold_stress'] * 0.4 +
        df['humidity_stress'] * 0.3 +
        df['pressure_stress'] * 0.3
    )
    
    # 体温調節ストレス（高温+高湿度）
    df['thermoregulatory_stress'] = (
        df['heat_stress'] * 0.5 +
        df['humidity_stress'] * 0.5
    )
    
    # 自律神経ストレス（気圧変動+温度変動）
    df['autonomic_stress'] = (
        df['pressure_stress'] * 0.6 +
        df['vtvf_temp_volatility'] * 0.4
    )
    
    # 4. 季節性を考慮した医学的ストレス
    
    # 冬期心血管ストレス
    df['winter_cardiovascular_stress'] = np.where(
        df['month'].isin([12, 1, 2]),
        df['cardiovascular_composite_stress'] * 1.5,  # 冬期は1.5倍
        df['cardiovascular_composite_stress']
    )
    
    # 夏期体温調節ストレス
    df['summer_thermoregulatory_stress'] = np.where(
        df['month'].isin([6, 7, 8]),
        df['thermoregulatory_stress'] * 1.3,  # 夏期は1.3倍
        df['thermoregulatory_stress']
    )
    
    # 5. 臨床的重症度分類
    
    # 軽度ストレス（注意喚起レベル）
    df['mild_stress'] = np.where(
        (df['cardiovascular_composite_stress'] > 0.5) | 
        (df['thermoregulatory_stress'] > 0.5),
        1, 0
    )
    
    # 中等度ストレス（医療機関受診推奨レベル）
    df['moderate_stress'] = np.where(
        (df['cardiovascular_composite_stress'] > 1.0) | 
        (df['thermoregulatory_stress'] > 1.0) |
        (df['autonomic_stress'] > 0.8),
        1, 0
    )
    
    # 重度ストレス（緊急医療対応レベル）
    df['severe_stress'] = np.where(
        (df['cardiovascular_composite_stress'] > 2.0) | 
        (df['thermoregulatory_stress'] > 2.0) |
        (df['autonomic_stress'] > 1.5),
        1, 0
    )
    
    return df

def analyze_clinical_weather_effects(diseases_data):
    """臨床的気象影響の分析"""
    print("\n=== 臨床的気象影響分析 ===")
    
    clinical_effects = {}
    
    for disease, data in diseases_data.items():
        disease_effects = {}
        
        # 疾患別の臨床的ストレス指標
        if disease == 'HF':
            stress_indicators = ['hf_cold_stress', 'hf_humidity_stress', 'cardiovascular_composite_stress']
        elif disease == 'AMI':
            stress_indicators = ['ami_cold_stress', 'ami_pressure_stress', 'cardiovascular_composite_stress']
        elif disease == 'VT_VF':
            stress_indicators = ['vtvf_humidity_stress', 'vtvf_temp_volatility', 'autonomic_stress']
        else:
            stress_indicators = ['cardiovascular_composite_stress', 'thermoregulatory_stress', 'autonomic_stress']
        
        for indicator in stress_indicators:
            if indicator in data.columns:
                # 高ストレス日の平均発症数
                high_stress_days = data[data[indicator] > data[indicator].quantile(0.9)]['incidence'].mean()
                # 低ストレス日の平均発症数
                low_stress_days = data[data[indicator] < data[indicator].quantile(0.1)]['incidence'].mean()
                # 通常日の平均発症数
                normal_days = data[(data[indicator] >= data[indicator].quantile(0.1)) & 
                                  (data[indicator] <= data[indicator].quantile(0.9))]['incidence'].mean()
                
                if normal_days > 0:
                    high_effect = high_stress_days / normal_days
                    low_effect = low_stress_days / normal_days
                    disease_effects[indicator] = {
                        'high_stress_effect': high_effect,
                        'low_stress_effect': low_effect,
                        'stress_range': data[indicator].max() - data[indicator].min()
                    }
        
        clinical_effects[disease] = disease_effects
    
    return clinical_effects

def analyze_severity_distribution(diseases_data):
    """重症度分布の分析"""
    print("\n=== 重症度分布分析 ===")
    
    severity_distribution = {}
    
    for disease, data in diseases_data.items():
        if 'mild_stress' in data.columns:
            severity_stats = {
                'mild_stress_days': data['mild_stress'].sum(),
                'moderate_stress_days': data['moderate_stress'].sum(),
                'severe_stress_days': data['severe_stress'].sum(),
                'total_days': len(data),
                'mild_stress_incidence': data[data['mild_stress'] == 1]['incidence'].mean(),
                'moderate_stress_incidence': data[data['moderate_stress'] == 1]['incidence'].mean(),
                'severe_stress_incidence': data[data['severe_stress'] == 1]['incidence'].mean(),
                'normal_incidence': data[(data['mild_stress'] == 0) & 
                                       (data['moderate_stress'] == 0) & 
                                       (data['severe_stress'] == 0)]['incidence'].mean()
            }
            severity_distribution[disease] = severity_stats
    
    # 重症度分布を可視化
    severity_df = pd.DataFrame(severity_distribution).T
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ストレス日数分布
    stress_days = severity_df[['mild_stress_days', 'moderate_stress_days', 'severe_stress_days']]
    stress_days.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Weather Stress Days by Disease')
    axes[0, 0].set_ylabel('Number of Days')
    axes[0, 0].legend(['Mild', 'Moderate', 'Severe'])
    
    # 発症率比較
    incidence_data = severity_df[['mild_stress_incidence', 'moderate_stress_incidence', 
                                 'severe_stress_incidence', 'normal_incidence']]
    incidence_data.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Incidence Rate by Stress Level')
    axes[0, 1].set_ylabel('Average Incidence')
    axes[0, 1].legend(['Mild Stress', 'Moderate Stress', 'Severe Stress', 'Normal'])
    
    # ストレスレベル別発症率増加
    mild_increase = (severity_df['mild_stress_incidence'] / severity_df['normal_incidence']).fillna(1)
    moderate_increase = (severity_df['moderate_stress_incidence'] / severity_df['normal_incidence']).fillna(1)
    severe_increase = (severity_df['severe_stress_incidence'] / severity_df['normal_incidence']).fillna(1)
    
    increase_data = pd.DataFrame({
        'Mild': mild_increase,
        'Moderate': moderate_increase,
        'Severe': severe_increase
    })
    increase_data.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Incidence Increase Ratio by Stress Level')
    axes[1, 0].set_ylabel('Increase Ratio (vs Normal)')
    
    # 疾患別ストレス感受性
    stress_sensitivity = severity_df[['mild_stress_days', 'moderate_stress_days', 'severe_stress_days']].sum(axis=1) / severity_df['total_days']
    stress_sensitivity.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Weather Stress Sensitivity by Disease')
    axes[1, 1].set_ylabel('Stress Days Ratio')
    
    plt.tight_layout()
    plt.savefig('visualizations/clinical_severity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 臨床的重症度分析図を保存: visualizations/clinical_severity_analysis.png")
    
    return severity_distribution

def create_medical_weather_report(clinical_effects, severity_distribution):
    """医学的気象報告書を作成"""
    print("\n=== 医学的気象報告書作成 ===")
    
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
        'medical_weather_definitions': {
            'cardiovascular_stress': {
                'definition': '寒冷、高湿度、気圧変動による心血管系への負荷',
                'clinical_mechanism': '血管収縮、血圧上昇、心拍数増加',
                'target_diseases': ['HF', 'AMI'],
                'thresholds': {
                    'mild': '0.5-1.0 (注意喚起レベル)',
                    'moderate': '1.0-2.0 (医療機関受診推奨)',
                    'severe': '>2.0 (緊急医療対応)'
                }
            },
            'thermoregulatory_stress': {
                'definition': '高温・高湿度による体温調節機能への負荷',
                'clinical_mechanism': '脱水、発汗困難、体温上昇',
                'target_diseases': ['HF', 'PE'],
                'thresholds': {
                    'mild': '0.5-1.0 (水分補給推奨)',
                    'moderate': '1.0-2.0 (涼しい場所での休息)',
                    'severe': '>2.0 (医療機関受診)'
                }
            },
            'autonomic_stress': {
                'definition': '気圧変動・温度変動による自律神経系への負荷',
                'clinical_mechanism': '交感神経亢進、不整脈誘発',
                'target_diseases': ['VT_VF', 'AMI'],
                'thresholds': {
                    'mild': '0.5-1.0 (安静推奨)',
                    'moderate': '1.0-1.5 (医療機関受診推奨)',
                    'severe': '>1.5 (緊急医療対応)'
                }
            }
        },
        'clinical_implications': {
            'preventive_medicine': [
                '気象予報に基づく疾患別リスク予測',
                '患者への気象注意喚起システム',
                '医療機関の受診調整システム'
            ],
            'emergency_medicine': [
                '気象条件に応じた救急医療体制調整',
                '重症患者の早期発見システム',
                '医療資源の効率的配分'
            ],
            'patient_education': [
                '疾患別気象注意点の啓蒙',
                '自己管理のための気象情報提供',
                '緊急時の対応指針提供'
            ]
        },
        'clinical_effects': convert_numpy(clinical_effects),
        'severity_distribution': convert_numpy(severity_distribution)
    }
    
    # レポートを保存
    with open('reports/medical_weather_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✓ 医学的気象報告書を保存: reports/medical_weather_analysis_report.json")
    return report

def main():
    """メイン実行関数"""
    print("=== 医学的気象ストレス分析開始 ===")
    
    # 1. データ読み込みと医学的気象ストレス指標作成
    diseases_data = {}
    
    # 東京全体データ
    tokyo_data = pd.read_csv('data/tokyo_weather_merged.csv')
    tokyo_data['date'] = pd.to_datetime(tokyo_data['date'])
    tokyo_data['disease'] = 'Tokyo_Total'
    tokyo_data['incidence'] = tokyo_data['people_tokyo']
    tokyo_data['year'] = tokyo_data['date'].dt.year
    tokyo_data['month'] = tokyo_data['date'].dt.month
    tokyo_data = define_medical_weather_stress(tokyo_data, 'weather')
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
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data = define_medical_weather_stress(data, '')
        diseases_data[disease] = data
    
    # 2. 臨床的気象影響分析
    clinical_effects = analyze_clinical_weather_effects(diseases_data)
    
    # 3. 重症度分布分析
    severity_distribution = analyze_severity_distribution(diseases_data)
    
    # 4. 医学的報告書作成
    report = create_medical_weather_report(clinical_effects, severity_distribution)
    
    print("\n=== 医学的気象分析完了 ===")
    print("生成されたファイル:")
    print("- visualizations/clinical_severity_analysis.png")
    print("- reports/medical_weather_analysis_report.json")
    
    print(f"\n医学的気象ストレス指標:")
    print(f"- 心血管ストレス: 寒冷、高湿度、気圧変動による負荷")
    print(f"- 体温調節ストレス: 高温・高湿度による体温調節機能への負荷")
    print(f"- 自律神経ストレス: 気圧変動・温度変動による自律神経系への負荷")
    print(f"- 重症度分類: 軽度(注意喚起)、中等度(受診推奨)、重度(緊急対応)")

if __name__ == "__main__":
    main() 