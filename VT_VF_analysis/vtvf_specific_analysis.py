import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import jpholiday
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_vtvf_data():
    """VT・VFデータを読み込み"""
    df = pd.read_csv('vtvf_weather_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['people_vtvf'] = df['people_vtvf'].fillna(0)
    return df

def analyze_vtvf_seasonality(df):
    """VT・VFの季節性分析"""
    print("=== VT・VF季節性分析 ===")
    
    # 月別のVT・VF発生数
    df['month'] = df['date'].dt.month
    monthly_vtvf = df.groupby('month')['people_vtvf'].agg(['sum', 'mean', 'count'])
    monthly_vtvf['avg_per_day'] = monthly_vtvf['sum'] / monthly_vtvf['count']
    
    print("月別VT・VF発生数:")
    print(monthly_vtvf)
    
    # 季節別分析
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    seasonal_vtvf = df.groupby('season')['people_vtvf'].agg(['sum', 'mean', 'count'])
    seasonal_vtvf['avg_per_day'] = seasonal_vtvf['sum'] / seasonal_vtvf['count']
    
    print("\n季節別VT・VF発生数:")
    print(seasonal_vtvf)
    
    return monthly_vtvf, seasonal_vtvf

def analyze_weather_impact_on_vtvf(df):
    """気象条件がVT・VFに与える影響分析"""
    print("\n=== 気象条件とVT・VFの関連性分析 ===")
    
    # 高VT・VF日（75%タイル以上）の特定
    vtvf_threshold = df['people_vtvf'].quantile(0.75)
    high_vtvf_days = df[df['people_vtvf'] >= vtvf_threshold]
    normal_days = df[df['people_vtvf'] < vtvf_threshold]
    
    print(f"高VT・VF日閾値: {vtvf_threshold:.1f}")
    print(f"高VT・VF日数: {len(high_vtvf_days)} ({len(high_vtvf_days)/len(df)*100:.1f}%)")
    
    # 気象条件の比較
    weather_cols = ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local', 'avg_wind_weather']
    
    print("\n気象条件比較（高VT・VF日 vs 通常日）:")
    for col in weather_cols:
        high_mean = high_vtvf_days[col].mean()
        normal_mean = normal_days[col].mean()
        diff = high_mean - normal_mean
        
        # t検定
        t_stat, p_value = stats.ttest_ind(high_vtvf_days[col], normal_days[col])
        
        print(f"{col}:")
        print(f"  高VT・VF日平均: {high_mean:.2f}")
        print(f"  通常日平均: {normal_mean:.2f}")
        print(f"  差: {diff:.2f}")
        print(f"  p値: {p_value:.4f}")
        print(f"  統計的有意性: {'あり' if p_value < 0.05 else 'なし'}")
        print()

def analyze_extreme_weather_impact(df):
    """極端な気象条件のVT・VFへの影響分析"""
    print("=== 極端な気象条件の影響分析 ===")
    
    # 極端な気象条件の定義
    df['extreme_temp'] = ((df['avg_temp_weather'] >= 30) | (df['avg_temp_weather'] <= 5)).astype(int)
    df['extreme_humidity'] = ((df['avg_humidity_weather'] >= 80) | (df['avg_humidity_weather'] <= 30)).astype(int)
    df['extreme_pressure'] = ((df['pressure_local'] >= 1020) | (df['pressure_local'] <= 990)).astype(int)
    df['extreme_wind'] = (df['avg_wind_weather'] >= 5).astype(int)
    
    # 気温変化の急激さ
    df['temp_change'] = df['avg_temp_weather'].diff().abs()
    df['rapid_temp_change'] = (df['temp_change'] >= 5).astype(int)
    
    # 気圧変化の急激さ
    df['pressure_change'] = df['pressure_local'].diff().abs()
    df['rapid_pressure_change'] = (df['pressure_change'] >= 10).astype(int)
    
    extreme_conditions = ['extreme_temp', 'extreme_humidity', 'extreme_pressure', 
                         'extreme_wind', 'rapid_temp_change', 'rapid_pressure_change']
    
    print("極端な気象条件とVT・VFの関連性:")
    for condition in extreme_conditions:
        condition_days = df[df[condition] == 1]
        normal_days = df[df[condition] == 0]
        
        if len(condition_days) > 0 and len(normal_days) > 0:
            condition_vtvf = condition_days['people_vtvf'].mean()
            normal_vtvf = normal_days['people_vtvf'].mean()
            
            # t検定
            t_stat, p_value = stats.ttest_ind(condition_days['people_vtvf'], normal_days['people_vtvf'])
            
            print(f"{condition}:")
            print(f"  条件日VT・VF平均: {condition_vtvf:.2f}")
            print(f"  通常日VT・VF平均: {normal_vtvf:.2f}")
            print(f"  差: {condition_vtvf - normal_vtvf:.2f}")
            print(f"  p値: {p_value:.4f}")
            print(f"  統計的有意性: {'あり' if p_value < 0.05 else 'なし'}")
            print()

def analyze_vtvf_patterns(df):
    """VT・VFの発生パターン分析"""
    print("=== VT・VF発生パターン分析 ===")
    
    # 曜日別分析
    df['dayofweek'] = df['date'].dt.dayofweek
    dow_vtvf = df.groupby('dayofweek')['people_vtvf'].mean()
    
    print("曜日別VT・VF発生数（平均）:")
    dow_names = ['月', '火', '水', '木', '金', '土', '日']
    for i, (dow, avg) in enumerate(dow_vtvf.items()):
        print(f"  {dow_names[i]}: {avg:.2f}")
    
    # 連続発生パターン
    df['vtvf_occurred'] = (df['people_vtvf'] > 0).astype(int)
    df['consecutive_vtvf'] = df['vtvf_occurred'].rolling(window=3).sum()
    
    consecutive_patterns = df['consecutive_vtvf'].value_counts().sort_index()
    print(f"\n連続発生パターン（3日間）:")
    for pattern, count in consecutive_patterns.items():
        print(f"  {pattern}日連続: {count}回")
    
    # 月曜日効果（週末後の影響）
    monday_vtvf = df[df['dayofweek'] == 0]['people_vtvf'].mean()
    other_days_vtvf = df[df['dayofweek'] != 0]['people_vtvf'].mean()
    
    print(f"\n月曜日効果:")
    print(f"  月曜日平均: {monday_vtvf:.2f}")
    print(f"  その他曜日平均: {other_days_vtvf:.2f}")
    print(f"  差: {monday_vtvf - other_days_vtvf:.2f}")

def create_vtvf_weather_correlation_analysis(df):
    """VT・VFと気象条件の相関分析"""
    print("=== VT・VFと気象条件の相関分析 ===")
    
    # 相関係数の計算
    weather_cols = ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local', 'avg_wind_weather']
    
    correlations = {}
    for col in weather_cols:
        corr = df['people_vtvf'].corr(df[col])
        correlations[col] = corr
        
        # スピアマン相関（非線形関係）
        spearman_corr = df['people_vtvf'].corr(df[col], method='spearman')
        
        print(f"{col}:")
        print(f"  ピアソン相関: {corr:.4f}")
        print(f"  スピアマン相関: {spearman_corr:.4f}")
        
        # 統計的有意性
        if abs(corr) > 0.1:
            print(f"  解釈: {'正の相関' if corr > 0 else '負の相関'}あり")
        else:
            print(f"  解釈: 相関なし")
        print()
    
    return correlations

def analyze_vtvf_risk_factors(df):
    """VT・VFリスク因子の特定"""
    print("=== VT・VFリスク因子分析 ===")
    
    # リスク因子の定義
    risk_factors = {
        'high_temp': df['avg_temp_weather'] >= 25,
        'low_temp': df['avg_temp_weather'] <= 5,
        'high_humidity': df['avg_humidity_weather'] >= 70,
        'low_pressure': df['pressure_local'] <= 1000,
        'high_wind': df['avg_wind_weather'] >= 4,
        'temp_change_large': df['avg_temp_weather'].diff().abs() >= 5,
        'pressure_change_large': df['pressure_local'].diff().abs() >= 10
    }
    
    print("リスク因子とVT・VF発生率:")
    for factor_name, factor_condition in risk_factors.items():
        risk_days = df[factor_condition]
        normal_days = df[~factor_condition]
        
        if len(risk_days) > 0 and len(normal_days) > 0:
            risk_vtvf_rate = risk_days['people_vtvf'].mean()
            normal_vtvf_rate = normal_days['people_vtvf'].mean()
            risk_ratio = risk_vtvf_rate / normal_vtvf_rate if normal_vtvf_rate > 0 else float('inf')
            
            print(f"{factor_name}:")
            print(f"  リスク日VT・VF率: {risk_vtvf_rate:.2f}")
            print(f"  通常日VT・VF率: {normal_vtvf_rate:.2f}")
            print(f"  リスク比: {risk_ratio:.2f}")
            print()

def create_vtvf_visualizations(df):
    """VT・VF分析の可視化"""
    print("=== VT・VF分析の可視化 ===")
    
    # プロット設定
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 月別VT・VF発生数
    monthly_vtvf = df.groupby(df['date'].dt.month)['people_vtvf'].mean()
    axes[0, 0].bar(monthly_vtvf.index, monthly_vtvf.values, color='skyblue')
    axes[0, 0].set_title('月別VT・VF発生数（平均）')
    axes[0, 0].set_xlabel('月')
    axes[0, 0].set_ylabel('VT・VF発生数')
    axes[0, 0].set_xticks(range(1, 13))
    
    # 2. 気温とVT・VFの関係
    temp_bins = pd.cut(df['avg_temp_weather'], bins=10)
    temp_vtvf = df.groupby(temp_bins)['people_vtvf'].mean()
    axes[0, 1].bar(range(len(temp_vtvf)), temp_vtvf.values, color='lightcoral')
    axes[0, 1].set_title('気温別VT・VF発生率')
    axes[0, 1].set_xlabel('気温区分')
    axes[0, 1].set_ylabel('VT・VF発生率')
    axes[0, 1].set_xticks(range(len(temp_vtvf)))
    axes[0, 1].set_xticklabels([f"{int(bin_.left)}-{int(bin_.right)}°C" for bin_ in temp_vtvf.index], rotation=45)
    
    # 3. 曜日別VT・VF発生数
    dow_vtvf = df.groupby(df['date'].dt.dayofweek)['people_vtvf'].mean()
    dow_names = ['月', '火', '水', '木', '金', '土', '日']
    axes[1, 0].bar(dow_names, dow_vtvf.values, color='lightgreen')
    axes[1, 0].set_title('曜日別VT・VF発生数（平均）')
    axes[1, 0].set_ylabel('VT・VF発生数')
    
    # 4. 気圧とVT・VFの関係
    pressure_bins = pd.cut(df['pressure_local'], bins=10)
    pressure_vtvf = df.groupby(pressure_bins)['people_vtvf'].mean()
    axes[1, 1].bar(range(len(pressure_vtvf)), pressure_vtvf.values, color='gold')
    axes[1, 1].set_title('気圧別VT・VF発生率')
    axes[1, 1].set_xlabel('気圧区分')
    axes[1, 1].set_ylabel('VT・VF発生率')
    axes[1, 1].set_xticks(range(len(pressure_vtvf)))
    axes[1, 1].set_xticklabels([f"{int(bin_.left)}-{int(bin_.right)}hPa" for bin_ in pressure_vtvf.index], rotation=45)
    
    plt.tight_layout()
    plt.savefig('vtvf_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("可視化結果を保存しました: vtvf_analysis_visualizations.png")

def main():
    """VT・VF特化分析のメイン実行"""
    print("=== VT・VF特化分析開始 ===")
    
    # データ読み込み
    df = load_vtvf_data()
    print(f"データ期間: {df['date'].min()} から {df['date'].max()}")
    print(f"総データ数: {len(df)}")
    print(f"VT・VF発生日数: {len(df[df['people_vtvf'] > 0])}")
    print()
    
    # 各分析の実行
    monthly_vtvf, seasonal_vtvf = analyze_vtvf_seasonality(df)
    analyze_weather_impact_on_vtvf(df)
    analyze_extreme_weather_impact(df)
    analyze_vtvf_patterns(df)
    correlations = create_vtvf_weather_correlation_analysis(df)
    analyze_vtvf_risk_factors(df)
    create_vtvf_visualizations(df)
    
    print("=== VT・VF特化分析完了 ===")
    
    # 結果の要約
    print("\n=== 分析結果要約 ===")
    print("1. 季節性: VT・VFの発生に季節性があるか確認")
    print("2. 気象影響: 極端な気象条件がVT・VFに与える影響")
    print("3. 発生パターン: 曜日や連続発生のパターン")
    print("4. リスク因子: VT・VFのリスクを高める気象条件")
    print("5. 相関分析: VT・VFと気象条件の相関関係")

if __name__ == "__main__":
    main() 