import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import jpholiday
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_af_data():
    """Afデータを読み込み"""
    print("=== Afデータ読み込み ===")
    df = pd.read_csv('af_weather_merged.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['hospitalization_date'] = df['date']
    
    print(f"データ期間: {df['date'].min()} から {df['date'].max()}")
    print(f"総データ数: {len(df)}")
    print(f"Af発生総数: {df['people_af'].sum()}")
    print(f"平均Af発生数/日: {df['people_af'].mean():.2f}")
    print(f"最大Af発生数/日: {df['people_af'].max()}")
    print(f"最小Af発生数/日: {df['people_af'].min()}")
    
    return df

def analyze_af_seasonality(df):
    """Afの季節性を分析"""
    print("\n=== Af季節性分析 ===")
    
    # 月別分析
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month')['people_af'].agg(['mean', 'sum', 'count']).round(2)
    monthly_stats.columns = ['平均Af発生数/日', '総Af発生数', 'データ日数']
    
    print("月別Af発生統計:")
    print(monthly_stats)
    
    # 季節別分析
    df['season'] = df['date'].dt.month.map({
        12: '冬', 1: '冬', 2: '冬',
        3: '春', 4: '春', 5: '春',
        6: '夏', 7: '夏', 8: '夏',
        9: '秋', 10: '秋', 11: '秋'
    })
    
    seasonal_stats = df.groupby('season')['people_af'].agg(['mean', 'sum', 'count']).round(2)
    seasonal_stats.columns = ['平均Af発生数/日', '総Af発生数', 'データ日数']
    
    print("\n季節別Af発生統計:")
    print(seasonal_stats)
    
    return monthly_stats, seasonal_stats

def analyze_weather_impact_on_af(df):
    """気象条件がAfに与える影響を分析"""
    print("\n=== 気象条件のAfへの影響分析 ===")
    
    # 75%タイルで高Af日を定義
    threshold = df['people_af'].quantile(0.75)
    high_af_days = df[df['people_af'] >= threshold]
    normal_days = df[df['people_af'] < threshold]
    
    print(f"高Af日定義: {threshold:.1f}人以上/日")
    print(f"高Af日数: {len(high_af_days)}日")
    print(f"通常日数: {len(normal_days)}日")
    
    # 気象要素の比較
    weather_cols = ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local', 
                   'avg_wind_weather', 'sunshine_hours_weather']
    
    print("\n気象条件の比較 (高Af日 vs 通常日):")
    for col in weather_cols:
        if col in df.columns:
            high_mean = high_af_days[col].mean()
            normal_mean = normal_days[col].mean()
            diff = high_mean - normal_mean
            t_stat, p_value = stats.ttest_ind(high_af_days[col], normal_days[col])
            
            print(f"{col}:")
            print(f"  高Af日平均: {high_mean:.2f}")
            print(f"  通常日平均: {normal_mean:.2f}")
            print(f"  差: {diff:.2f}")
            print(f"  p値: {p_value:.4f}")
            print(f"  有意性: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
            print()

def analyze_extreme_weather_impact(df):
    """極端な気象条件のAfへの影響を分析"""
    print("\n=== 極端な気象条件のAfへの影響分析 ===")
    
    # 極端な気象条件の定義
    extreme_conditions = {
        '極端な高温': df['avg_temp_weather'] >= 30,
        '極端な低温': df['avg_temp_weather'] <= 0,
        '極端な高湿度': df['avg_humidity_weather'] >= 80,
        '極端な低湿度': df['avg_humidity_weather'] <= 30,
        '極端な高気圧': df['pressure_local'] >= 1020,
        '極端な低気圧': df['pressure_local'] <= 1000,
        '強風': df['avg_wind_weather'] >= 10
    }
    
    threshold = df['people_af'].quantile(0.75)
    
    for condition_name, condition_mask in extreme_conditions.items():
        if condition_mask.sum() > 0:
            extreme_days = df[condition_mask]
            normal_days = df[~condition_mask]
            
            extreme_af_mean = extreme_days['people_af'].mean()
            normal_af_mean = normal_days['people_af'].mean()
            
            high_af_extreme = (extreme_days['people_af'] >= threshold).sum()
            high_af_normal = (normal_days['people_af'] >= threshold).sum()
            
            total_extreme = len(extreme_days)
            total_normal = len(normal_days)
            
            if total_extreme > 0 and total_normal > 0:
                extreme_rate = high_af_extreme / total_extreme
                normal_rate = high_af_normal / total_normal
                
                # カイ二乗検定
                observed = np.array([[high_af_extreme, total_extreme - high_af_extreme],
                                   [high_af_normal, total_normal - high_af_normal]])
                chi2, p_value = stats.chi2_contingency(observed)[:2]
                
                print(f"{condition_name}:")
                print(f"  極端条件日数: {total_extreme}日")
                print(f"  通常条件日数: {total_normal}日")
                print(f"  極端条件でのAf平均: {extreme_af_mean:.2f}")
                print(f"  通常条件でのAf平均: {normal_af_mean:.2f}")
                print(f"  極端条件での高Af率: {extreme_rate:.3f}")
                print(f"  通常条件での高Af率: {normal_rate:.3f}")
                print(f"  p値: {p_value:.4f}")
                print(f"  有意性: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
                print()

def analyze_af_patterns(df):
    """Afのパターンを分析"""
    print("\n=== Afパターン分析 ===")
    
    # 曜日別分析
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    
    dow_stats = df.groupby('day_name')['people_af'].agg(['mean', 'sum', 'count']).round(2)
    dow_stats.columns = ['平均Af発生数/日', '総Af発生数', 'データ日数']
    
    print("曜日別Af発生統計:")
    print(dow_stats)
    
    # 月の前半/後半分析
    df['month_half'] = df['date'].dt.day.apply(lambda x: '前半' if x <= 15 else '後半')
    half_stats = df.groupby('month_half')['people_af'].agg(['mean', 'sum', 'count']).round(2)
    half_stats.columns = ['平均Af発生数/日', '総Af発生数', 'データ日数']
    
    print("\n月の前半/後半別Af発生統計:")
    print(half_stats)

def create_af_weather_correlation_analysis(df):
    """Afと気象条件の相関分析"""
    print("\n=== Afと気象条件の相関分析 ===")
    
    weather_cols = ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local', 
                   'avg_wind_weather', 'sunshine_hours_weather']
    
    correlations = {}
    for col in weather_cols:
        if col in df.columns:
            # ピアソン相関
            pearson_corr, pearson_p = stats.pearsonr(df['people_af'], df[col])
            # スピアマン相関
            spearman_corr, spearman_p = stats.spearmanr(df['people_af'], df[col])
            
            correlations[col] = {
                'pearson': (pearson_corr, pearson_p),
                'spearman': (spearman_corr, spearman_p)
            }
            
            print(f"{col}:")
            print(f"  ピアソン相関: {pearson_corr:.4f} (p={pearson_p:.4f})")
            print(f"  スピアマン相関: {spearman_corr:.4f} (p={spearman_p:.4f})")
            print()
    
    return correlations

def analyze_af_risk_factors(df):
    """Afのリスク要因を分析"""
    print("\n=== Afリスク要因分析 ===")
    
    # 複合気象条件の分析
    print("複合気象条件のAfへの影響:")
    
    # 高温高湿度
    hot_humid = (df['avg_temp_weather'] >= 25) & (df['avg_humidity_weather'] >= 70)
    if hot_humid.sum() > 0:
        hot_humid_af = df[hot_humid]['people_af'].mean()
        other_af = df[~hot_humid]['people_af'].mean()
        print(f"高温高湿度日: {hot_humid.sum()}日, Af平均: {hot_humid_af:.2f} vs その他: {other_af:.2f}")
    
    # 低温低湿度
    cold_dry = (df['avg_temp_weather'] <= 5) & (df['avg_humidity_weather'] <= 50)
    if cold_dry.sum() > 0:
        cold_dry_af = df[cold_dry]['people_af'].mean()
        other_af = df[~cold_dry]['people_af'].mean()
        print(f"低温低湿度日: {cold_dry.sum()}日, Af平均: {cold_dry_af:.2f} vs その他: {other_af:.2f}")
    
    # 気圧変動
    df['pressure_change'] = df['pressure_local'].diff()
    high_pressure_change = abs(df['pressure_change']) >= 5
    if high_pressure_change.sum() > 0:
        high_change_af = df[high_pressure_change]['people_af'].mean()
        low_change_af = df[~high_pressure_change]['people_af'].mean()
        print(f"気圧変動大: {high_pressure_change.sum()}日, Af平均: {high_change_af:.2f} vs その他: {low_change_af:.2f}")

def create_af_visualizations(df):
    """Afの可視化を作成"""
    print("\n=== Af可視化作成 ===")
    
    # 月別Af発生数の可視化
    plt.figure(figsize=(15, 10))
    
    # 1. 月別Af発生数
    plt.subplot(2, 3, 1)
    monthly_af = df.groupby(df['date'].dt.month)['people_af'].mean()
    monthly_af.plot(kind='bar', color='skyblue')
    plt.title('月別平均Af発生数')
    plt.xlabel('月')
    plt.ylabel('平均Af発生数/日')
    plt.xticks(range(12), ['1月', '2月', '3月', '4月', '5月', '6月', 
                           '7月', '8月', '9月', '10月', '11月', '12月'])
    
    # 2. 曜日別Af発生数
    plt.subplot(2, 3, 2)
    dow_af = df.groupby('day_name')['people_af'].mean()
    dow_af.plot(kind='bar', color='lightcoral')
    plt.title('曜日別平均Af発生数')
    plt.xlabel('曜日')
    plt.ylabel('平均Af発生数/日')
    
    # 3. 気温とAfの関係
    plt.subplot(2, 3, 3)
    plt.scatter(df['avg_temp_weather'], df['people_af'], alpha=0.5, color='orange')
    plt.xlabel('平均気温 (°C)')
    plt.ylabel('Af発生数')
    plt.title('気温とAf発生数の関係')
    
    # 4. 湿度とAfの関係
    plt.subplot(2, 3, 4)
    plt.scatter(df['avg_humidity_weather'], df['people_af'], alpha=0.5, color='green')
    plt.xlabel('平均湿度 (%)')
    plt.ylabel('Af発生数')
    plt.title('湿度とAf発生数の関係')
    
    # 5. 気圧とAfの関係
    plt.subplot(2, 3, 5)
    plt.scatter(df['pressure_local'], df['people_af'], alpha=0.5, color='purple')
    plt.xlabel('気圧 (hPa)')
    plt.ylabel('Af発生数')
    plt.title('気圧とAf発生数の関係')
    
    # 6. 季節別Af発生数
    plt.subplot(2, 3, 6)
    seasonal_af = df.groupby('season')['people_af'].mean()
    seasonal_af.plot(kind='bar', color='gold')
    plt.title('季節別平均Af発生数')
    plt.xlabel('季節')
    plt.ylabel('平均Af発生数/日')
    
    plt.tight_layout()
    plt.savefig('af_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可視化を保存しました: af_analysis_visualizations.png")

def main():
    """メイン関数"""
    print("=== Af特化分析開始 ===")
    
    # データ読み込み
    df = load_af_data()
    
    # 季節性分析
    monthly_stats, seasonal_stats = analyze_af_seasonality(df)
    
    # 気象影響分析
    analyze_weather_impact_on_af(df)
    
    # 極端気象影響分析
    analyze_extreme_weather_impact(df)
    
    # パターン分析
    analyze_af_patterns(df)
    
    # 相関分析
    correlations = create_af_weather_correlation_analysis(df)
    
    # リスク要因分析
    analyze_af_risk_factors(df)
    
    # 可視化
    create_af_visualizations(df)
    
    print("\n=== Af特化分析完了 ===")

if __name__ == "__main__":
    main() 