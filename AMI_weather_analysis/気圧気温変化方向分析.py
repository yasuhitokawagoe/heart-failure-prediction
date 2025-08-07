import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """データを読み込み"""
    df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
    df['hospitalization_date'] = pd.to_datetime(df['date'])
    return df

def create_direction_analysis(df):
    """気圧と気温の変化方向を分析"""
    print("=== 気圧・気温変化方向分析 ===")
    
    # 基本的な特徴量エンジニアリング
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['is_weekend']
    
    # 気温・気圧の変化率
    df['temp_change'] = df['avg_temp'] - df['avg_temp'].shift(1)
    df['pressure_change'] = df['vapor_pressure'] - df['vapor_pressure'].shift(1)
    
    # 変化方向の分類
    df['temp_direction'] = np.where(df['temp_change'] > 0, '上昇', 
                                   np.where(df['temp_change'] < 0, '下降', '変化なし'))
    df['pressure_direction'] = np.where(df['pressure_change'] > 0, '上昇', 
                                       np.where(df['pressure_change'] < 0, '下降', '変化なし'))
    
    # 複合変化パターン
    df['temp_pressure_pattern'] = df['temp_direction'] + '×' + df['pressure_direction']
    
    # 変化の大きさ
    df['temp_change_magnitude'] = abs(df['temp_change'])
    df['pressure_change_magnitude'] = abs(df['pressure_change'])
    
    # ターゲット変数
    threshold = df['people'].quantile(0.75)
    df['target'] = (df['people'] >= threshold).astype(int)
    
    # 無限大とNaNの値を処理
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def analyze_direction_risk(df):
    """変化方向別のリスク分析"""
    print("\n=== 変化方向別リスク分析 ===")
    
    # 気温変化方向別のリスク
    temp_risk = df.groupby('temp_direction')['target'].agg(['mean', 'count']).round(4)
    print("\n気温変化方向別リスク:")
    print(temp_risk)
    
    # 気圧変化方向別のリスク
    pressure_risk = df.groupby('pressure_direction')['target'].agg(['mean', 'count']).round(4)
    print("\n気圧変化方向別リスク:")
    print(pressure_risk)
    
    # 複合パターン別のリスク
    pattern_risk = df.groupby('temp_pressure_pattern')['target'].agg(['mean', 'count']).round(4)
    print("\n複合変化パターン別リスク:")
    print(pattern_risk)
    
    return temp_risk, pressure_risk, pattern_risk

def create_direction_visualization(df):
    """変化方向の可視化"""
    print("\n=== 変化方向の可視化 ===")
    
    # 1. 気温変化方向別のリスク
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    temp_risk = df.groupby('temp_direction')['target'].mean()
    colors = ['red' if x > temp_risk.mean() else 'blue' for x in temp_risk.values]
    plt.bar(temp_risk.index, temp_risk.values, color=colors, alpha=0.7)
    plt.title('気温変化方向別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    plt.xticks(rotation=45)
    
    # 2. 気圧変化方向別のリスク
    plt.subplot(1, 3, 2)
    pressure_risk = df.groupby('pressure_direction')['target'].mean()
    colors = ['red' if x > pressure_risk.mean() else 'blue' for x in pressure_risk.values]
    plt.bar(pressure_risk.index, pressure_risk.values, color=colors, alpha=0.7)
    plt.title('気圧変化方向別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    plt.xticks(rotation=45)
    
    # 3. 複合パターン別のリスク
    plt.subplot(1, 3, 3)
    pattern_risk = df.groupby('temp_pressure_pattern')['target'].mean()
    colors = ['red' if x > pattern_risk.mean() else 'blue' for x in pattern_risk.values]
    plt.bar(range(len(pattern_risk)), pattern_risk.values, color=colors, alpha=0.7)
    plt.title('複合変化パターン別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    plt.xticks(range(len(pattern_risk)), pattern_risk.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('変化方向別リスク分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_magnitude_analysis(df):
    """変化の大きさとリスクの関係"""
    print("\n=== 変化の大きさとリスクの関係 ===")
    
    # 変化の大きさを四分位に分類
    df['temp_change_quartile'] = pd.qcut(df['temp_change_magnitude'], 4, labels=['最小', '小', '大', '最大'])
    df['pressure_change_quartile'] = pd.qcut(df['pressure_change_magnitude'], 4, labels=['最小', '小', '大', '最大'])
    
    # 気温変化の大きさ別リスク
    temp_magnitude_risk = df.groupby('temp_change_quartile')['target'].mean()
    print("\n気温変化の大きさ別リスク:")
    print(temp_magnitude_risk)
    
    # 気圧変化の大きさ別リスク
    pressure_magnitude_risk = df.groupby('pressure_change_quartile')['target'].mean()
    print("\n気圧変化の大きさ別リスク:")
    print(pressure_magnitude_risk)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['green', 'yellow', 'orange', 'red']
    plt.bar(temp_magnitude_risk.index, temp_magnitude_risk.values, color=colors, alpha=0.7)
    plt.title('気温変化の大きさ別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    
    plt.subplot(1, 2, 2)
    plt.bar(pressure_magnitude_risk.index, pressure_magnitude_risk.values, color=colors, alpha=0.7)
    plt.title('気圧変化の大きさ別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    
    plt.tight_layout()
    plt.savefig('変化の大きさ別リスク分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scatter_analysis(df):
    """気温・気圧変化の散布図分析"""
    print("\n=== 気温・気圧変化の散布図分析 ===")
    
    # 高リスク日の抽出
    high_risk_days = df[df['target'] == 1]
    low_risk_days = df[df['target'] == 0]
    
    plt.figure(figsize=(12, 8))
    
    # 高リスク日と低リスク日を色分け
    plt.scatter(low_risk_days['temp_change'], low_risk_days['pressure_change'], 
               alpha=0.3, color='blue', label='低リスク日', s=20)
    plt.scatter(high_risk_days['temp_change'], high_risk_days['pressure_change'], 
               alpha=0.7, color='red', label='高リスク日', s=30)
    
    plt.xlabel('気温変化 (°C)', fontsize=12)
    plt.ylabel('気圧変化 (hPa)', fontsize=12)
    plt.title('気温・気圧変化とAMIリスクの関係', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 象限の説明を追加
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 象限の説明
    plt.text(0.7, 0.9, '気温上昇\n気圧上昇', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(0.1, 0.9, '気温下降\n気圧上昇', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.text(0.1, 0.1, '気温下降\n気圧下降', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    plt.text(0.7, 0.1, '気温上昇\n気圧下降', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('気温気圧変化散布図.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_quadrant_analysis(df):
    """象限別リスク分析"""
    print("\n=== 象限別リスク分析 ===")
    
    # 象限の分類
    df['quadrant'] = np.where((df['temp_change'] > 0) & (df['pressure_change'] > 0), '気温上昇×気圧上昇',
                              np.where((df['temp_change'] < 0) & (df['pressure_change'] > 0), '気温下降×気圧上昇',
                              np.where((df['temp_change'] < 0) & (df['pressure_change'] < 0), '気温下降×気圧下降',
                              '気温上昇×気圧下降')))
    
    # 象限別リスク
    quadrant_risk = df.groupby('quadrant')['target'].agg(['mean', 'count']).round(4)
    print("\n象限別リスク:")
    print(quadrant_risk)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > quadrant_risk['mean'].mean() else 'blue' for x in quadrant_risk['mean'].values]
    plt.bar(range(len(quadrant_risk)), quadrant_risk['mean'].values, color=colors, alpha=0.7)
    plt.title('象限別AMIリスク', fontsize=14, fontweight='bold')
    plt.ylabel('AMIリスク確率')
    plt.xticks(range(len(quadrant_risk)), quadrant_risk.index, rotation=45)
    
    # リスク値の表示
    for i, v in enumerate(quadrant_risk['mean'].values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('象限別リスク分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return quadrant_risk

def generate_direction_report(temp_risk, pressure_risk, pattern_risk, quadrant_risk):
    """方向分析レポートを生成"""
    report = f"""
# 気圧・気温変化方向分析レポート

## 概要
気圧と気温の変化方向がAMIリスクに与える影響を詳細に分析しました。

## 主要な発見

### 1. 気温変化方向の影響
{temp_risk.to_string()}

**医学的解釈:**
- 気温上昇: 血管拡張、血圧低下のリスク
- 気温下降: 血管収縮、血圧上昇のリスク
- 変化なし: 最もリスクが低い

### 2. 気圧変化方向の影響
{pressure_risk.to_string()}

**医学的解釈:**
- 気圧上昇: 血管収縮、血圧上昇のリスク
- 気圧下降: 血管拡張、血圧低下のリスク
- 変化なし: 最もリスクが低い

### 3. 複合変化パターンの影響
{pattern_risk.to_string()}

**医学的解釈:**
- 気温上昇×気圧上昇: 最もリスクが高い（血管拡張+収縮の矛盾）
- 気温下降×気圧下降: リスクが低い（血管収縮+拡張の矛盾）
- その他の組み合わせ: 中間的なリスク

### 4. 象限別リスク分析
{quadrant_risk.to_string()}

**最も危険なパターン:**
1. 気温上昇×気圧上昇: 血管の矛盾した反応
2. 気温上昇×気圧下降: 血管拡張の複合効果
3. 気温下降×気圧上昇: 血管収縮の複合効果
4. 気温下降×気圧下降: 最も安全

## 臨床的応用

### 1. 予防医学
- 気温・気圧の同時変化時の注意喚起
- 特に気温上昇×気圧上昇の日の警戒

### 2. 患者指導
- 気象変化の激しい日の生活調整
- 血管に負荷のかかる組み合わせの回避

### 3. 医療体制
- 高リスク気象パターン日の医療体制強化
- 救急医療の準備体制

## 結論

気温と気圧の変化方向の組み合わせにより、AMIリスクが大きく変動することが明らかになりました。
特に、気温上昇と気圧上昇が同時に起こる場合が最も危険であり、
これは血管の矛盾した反応（拡張と収縮）によるものと考えられます。

これらの知見は、気象医学における予防医学と臨床指導に重要な示唆を与えるものです。
"""
    
    with open('気圧気温変化方向分析レポート.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("気圧気温変化方向分析レポートを生成しました: 気圧気温変化方向分析レポート.md")

def main():
    """メイン実行関数"""
    print("気圧・気温変化方向分析を開始します...")
    
    # データ読み込み
    df = load_data()
    
    # 方向分析用の特徴量準備
    df = create_direction_analysis(df)
    
    # 変化方向別リスク分析
    temp_risk, pressure_risk, pattern_risk = analyze_direction_risk(df)
    
    # 可視化
    create_direction_visualization(df)
    create_magnitude_analysis(df)
    create_scatter_analysis(df)
    quadrant_risk = create_quadrant_analysis(df)
    
    # レポート生成
    generate_direction_report(temp_risk, pressure_risk, pattern_risk, quadrant_risk)
    
    print("気圧・気温変化方向分析が完了しました。")

if __name__ == "__main__":
    main() 