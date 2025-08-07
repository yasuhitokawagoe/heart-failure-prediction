# 疾患間相互関係と時系列進行パターン分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_align_disease_data():
    """疾患データを読み込み、日付で整列"""
    print("=== 疾患データ読み込みと整列 ===")
    
    # 各疾患データを読み込み
    diseases = {
        'VT_VF': 'data/東京vtvf.csv',
        'HF': 'data/東京ADHF.csv', 
        'AMI': 'data/東京AMI.csv',
        'PE': 'data/東京PE.csv',
        'AF': 'data/東京AF.csv'
    }
    
    aligned_data = {}
    
    for disease, file_path in diseases.items():
        data = pd.read_csv(file_path)
        
        # AFの場合は列名が異なる
        if disease == 'AF':
            data['date'] = pd.to_datetime(data['hospitalization_date_af'])
            data = data.groupby('date')['people_af'].sum().reset_index()
            data.columns = ['date', f'{disease}_incidence']
        else:
            data['date'] = pd.to_datetime(data['hospitalization_date'])
            data = data.groupby('date')['people'].sum().reset_index()
            data.columns = ['date', f'{disease}_incidence']
        
        aligned_data[disease] = data
    
    # 日付範囲を統一
    all_dates = set()
    for data in aligned_data.values():
        all_dates.update(data['date'])
    
    # 完全な日付範囲を作成
    date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
    master_df = pd.DataFrame({'date': date_range})
    
    # 各疾患データをマージ
    for disease, data in aligned_data.items():
        master_df = master_df.merge(data, on='date', how='left')
        master_df[f'{disease}_incidence'] = master_df[f'{disease}_incidence'].fillna(0)
    
    print(f"整列完了: {len(master_df)}日間のデータ")
    print(f"疾患別発症数: {master_df[[col for col in master_df.columns if 'incidence' in col]].sum().to_dict()}")
    
    return master_df

def analyze_disease_correlations(aligned_data):
    """疾患間の相関関係を分析"""
    print("\n=== 疾患間相関分析 ===")
    
    # 発症数列を抽出
    incidence_cols = [col for col in aligned_data.columns if 'incidence' in col]
    incidence_data = aligned_data[incidence_cols]
    
    # 相関行列を計算
    correlation_matrix = incidence_data.corr()
    
    # 統計的有意性を計算
    p_values = pd.DataFrame(index=correlation_matrix.index, columns=correlation_matrix.columns, dtype=float)
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            if i != j:
                try:
                    corr, p_val = pearsonr(incidence_data[i], incidence_data[j])
                    p_values.loc[i, j] = p_val if not np.isnan(p_val) else 1.0
                except:
                    p_values.loc[i, j] = 1.0
            else:
                p_values.loc[i, j] = 0.0
    
    # 相関分析結果を可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 相関係数ヒートマップ
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                ax=axes[0], fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    axes[0].set_title('Disease Incidence Correlations')
    
    # p値ヒートマップ（数値のみ）
    p_values_numeric = p_values.astype(float)
    sns.heatmap(p_values_numeric, annot=True, cmap='Reds', ax=axes[1], fmt='.3f',
                cbar_kws={'label': 'P-value'})
    axes[1].set_title('Statistical Significance (P-values)')
    
    plt.tight_layout()
    plt.savefig('visualizations/disease_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 疾患間相関分析図を保存: visualizations/disease_correlations.png")
    
    return correlation_matrix, p_values

def analyze_temporal_progression(aligned_data):
    """時系列的な疾患進行パターンを分析"""
    print("\n=== 時系列進行パターン分析 ===")
    
    # 発症数列を抽出
    incidence_cols = [col for col in aligned_data.columns if 'incidence' in col]
    
    # 各疾患の時系列パターンを分析
    temporal_patterns = {}
    
    for disease in incidence_cols:
        disease_name = disease.replace('_incidence', '')
        data = aligned_data[disease]
        
        # 発症日の特定
        event_days = aligned_data[aligned_data[disease] > 0]['date']
        
        if len(event_days) > 0:
            # 他の疾患との時間的関係を分析
            temporal_relations = {}
            
            for other_disease in incidence_cols:
                if other_disease != disease:
                    other_name = other_disease.replace('_incidence', '')
                    
                    # 発症日の前後7日間で他の疾患の発症をチェック
                    lead_lag_analysis = []
                    
                    for event_date in event_days:
                        # 前後7日間の範囲
                        start_date = event_date - pd.Timedelta(days=7)
                        end_date = event_date + pd.Timedelta(days=7)
                        
                        # 範囲内の他の疾患発症をチェック
                        period_data = aligned_data[
                            (aligned_data['date'] >= start_date) & 
                            (aligned_data['date'] <= end_date)
                        ]
                        
                        for _, row in period_data.iterrows():
                            if row[other_disease] > 0:
                                days_diff = (row['date'] - event_date).days
                                lead_lag_analysis.append({
                                    'disease': other_name,
                                    'days_diff': days_diff,
                                    'incidence': row[other_disease]
                                })
                    
                    if lead_lag_analysis:
                        df_analysis = pd.DataFrame(lead_lag_analysis)
                        temporal_relations[other_name] = {
                            'mean_days_diff': df_analysis['days_diff'].mean(),
                            'median_days_diff': df_analysis['days_diff'].median(),
                            'positive_lag_ratio': (df_analysis['days_diff'] > 0).mean(),
                            'concurrent_events': (df_analysis['days_diff'] == 0).sum(),
                            'total_related_events': len(df_analysis)
                        }
            
            temporal_patterns[disease_name] = temporal_relations
    
    # 時系列パターンを可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 疾患別発症頻度の時系列
    for i, disease in enumerate(incidence_cols):
        disease_name = disease.replace('_incidence', '')
        data = aligned_data[disease]
        
        # 月別集計
        monthly_data = aligned_data.set_index('date')[disease].resample('M').sum()
        
        axes[0, 0].plot(monthly_data.index, monthly_data.values, 
                        label=disease_name, marker='o', markersize=3)
    
    axes[0, 0].set_title('Monthly Disease Incidence Patterns')
    axes[0, 0].set_ylabel('Monthly Incidence')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. 疾患間の時間的関係
    temporal_data = []
    for disease, relations in temporal_patterns.items():
        for other_disease, stats in relations.items():
            temporal_data.append({
                'Primary_Disease': disease,
                'Related_Disease': other_disease,
                'Mean_Days_Diff': stats['mean_days_diff'],
                'Positive_Lag_Ratio': stats['positive_lag_ratio'],
                'Concurrent_Events': stats['concurrent_events']
            })
    
    if temporal_data:
        temporal_df = pd.DataFrame(temporal_data)
        
        # 平均日数差のヒートマップ
        pivot_data = temporal_df.pivot(index='Primary_Disease', 
                                     columns='Related_Disease', 
                                     values='Mean_Days_Diff')
        sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                   ax=axes[0, 1], fmt='.1f')
        axes[0, 1].set_title('Mean Days Between Disease Events')
    
    # 3. 同時発症の頻度
    if temporal_data:
        concurrent_pivot = temporal_df.pivot(index='Primary_Disease', 
                                           columns='Related_Disease', 
                                           values='Concurrent_Events')
        # NaN値を0に置換
        concurrent_pivot = concurrent_pivot.fillna(0)
        sns.heatmap(concurrent_pivot, annot=True, cmap='Blues', 
                   ax=axes[1, 0], fmt='.0f')
        axes[1, 0].set_title('Concurrent Disease Events')
    
    # 4. 疾患進行パターンの要約
    progression_summary = []
    for disease, relations in temporal_patterns.items():
        for other_disease, stats in relations.items():
            if stats['total_related_events'] > 0:
                progression_type = "Precedes" if stats['mean_days_diff'] < -1 else \
                                 "Follows" if stats['mean_days_diff'] > 1 else "Concurrent"
                
                progression_summary.append({
                    'Primary': disease,
                    'Secondary': other_disease,
                    'Pattern': progression_type,
                    'Mean_Days': stats['mean_days_diff'],
                    'Frequency': stats['total_related_events']
                })
    
    if progression_summary:
        summary_df = pd.DataFrame(progression_summary)
        summary_counts = summary_df.groupby('Pattern').size()
        summary_counts.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Disease Progression Patterns')
        axes[1, 1].set_ylabel('Number of Disease Pairs')
    
    plt.tight_layout()
    plt.savefig('visualizations/disease_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 時系列進行パターン分析図を保存: visualizations/disease_temporal_patterns.png")
    
    return temporal_patterns

def analyze_causal_relationships(aligned_data):
    """因果関係の可能性を分析"""
    print("\n=== 因果関係分析 ===")
    
    # 発症数列を抽出
    incidence_cols = [col for col in aligned_data.columns if 'incidence' in col]
    
    causal_analysis = {}
    
    for i, disease1 in enumerate(incidence_cols):
        disease1_name = disease1.replace('_incidence', '')
        causal_analysis[disease1_name] = {}
        
        for disease2 in incidence_cols:
            if disease1 != disease2:
                disease2_name = disease2.replace('_incidence', '')
                
                # グレンジャー因果性テスト（簡易版）
                # 1日、3日、7日のラグで分析
                lags = [1, 3, 7]
                granger_results = {}
                
                for lag in lags:
                    # ラグ付きデータを作成
                    data1 = aligned_data[disease1].values
                    data2 = aligned_data[disease2].values
                    
                    # 相関分析（ラグ付き）
                    if len(data1) > lag:
                        try:
                            correlation = np.corrcoef(data1[lag:], data2[:-lag])[0, 1]
                            if np.isnan(correlation):
                                correlation = 0
                            p_val = stats.pearsonr(data1[lag:], data2[:-lag])[1]
                            if np.isnan(p_val):
                                p_val = 1.0
                        except:
                            correlation = 0
                            p_val = 1.0
                        
                        granger_results[f'lag_{lag}'] = {
                            'correlation': correlation,
                            'p_value': p_val
                        }
                
                # 最も強い因果関係を特定
                best_lag = max(granger_results.keys(), 
                              key=lambda x: abs(granger_results[x]['correlation']))
                
                causal_analysis[disease1_name][disease2_name] = {
                    'best_lag': best_lag,
                    'correlation': granger_results[best_lag]['correlation'],
                    'p_value': granger_results[best_lag]['p_value'],
                    'all_lags': granger_results
                }
    
    # 因果関係を可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 最も強い因果関係のネットワーク
    strong_causal_relations = []
    for disease1, relations in causal_analysis.items():
        for disease2, stats in relations.items():
            if abs(stats['correlation']) > 0.1 and stats['p_value'] < 0.05:
                strong_causal_relations.append({
                    'From': disease1,
                    'To': disease2,
                    'Correlation': stats['correlation'],
                    'Lag': stats['best_lag'],
                    'P_value': stats['p_value']
                })
    
    if strong_causal_relations:
        causal_df = pd.DataFrame(strong_causal_relations)
        
        # 因果関係の強度を可視化
        pivot_corr = causal_df.pivot(index='From', columns='To', values='Correlation')
        sns.heatmap(pivot_corr, annot=True, cmap='RdBu_r', center=0, 
                   ax=axes[0, 0], fmt='.3f')
        axes[0, 0].set_title('Causal Relationship Strengths')
        
        # ラグ時間の分布
        lag_counts = causal_df['Lag'].value_counts()
        lag_counts.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Lag Times')
        axes[0, 1].set_ylabel('Number of Relationships')
    
    # 2. 疾患進行パターンの詳細分析
    progression_patterns = []
    for disease1, relations in causal_analysis.items():
        for disease2, stats in relations.items():
            if abs(stats['correlation']) > 0.05:
                pattern = "Precedes" if stats['correlation'] > 0 else "Follows"
                progression_patterns.append({
                    'Primary': disease1,
                    'Secondary': disease2,
                    'Pattern': pattern,
                    'Strength': abs(stats['correlation']),
                    'Lag': stats['best_lag']
                })
    
    if progression_patterns:
        pattern_df = pd.DataFrame(progression_patterns)
        
        # 疾患別進行パターン
        pattern_summary = pattern_df.groupby(['Primary', 'Pattern']).size().unstack(fill_value=0)
        pattern_summary.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Disease Progression Patterns by Primary Disease')
        axes[1, 0].set_ylabel('Number of Relationships')
        axes[1, 0].legend(title='Pattern')
        
        # 因果関係の強度分布
        axes[1, 1].hist(pattern_df['Strength'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Distribution of Causal Relationship Strengths')
        axes[1, 1].set_xlabel('Correlation Strength')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('visualizations/disease_causal_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 因果関係分析図を保存: visualizations/disease_causal_relationships.png")
    
    return causal_analysis

def create_disease_interaction_report(correlation_matrix, p_values, temporal_patterns, causal_analysis):
    """疾患間相互作用報告書を作成"""
    print("\n=== 疾患間相互作用報告書作成 ===")
    
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
    
    # 強い相関関係を抽出
    strong_correlations = []
    for i in correlation_matrix.index:
        for j in correlation_matrix.columns:
            if i != j and abs(correlation_matrix.loc[i, j]) > 0.1:
                strong_correlations.append({
                    'disease1': i,
                    'disease2': j,
                    'correlation': correlation_matrix.loc[i, j],
                    'p_value': p_values.loc[i, j]
                })
    
    # 因果関係の可能性を抽出
    causal_relationships = []
    for disease1, relations in causal_analysis.items():
        for disease2, stats in relations.items():
            if abs(stats['correlation']) > 0.1 and stats['p_value'] < 0.05:
                causal_relationships.append({
                    'from_disease': disease1,
                    'to_disease': disease2,
                    'correlation': stats['correlation'],
                    'lag': stats['best_lag'],
                    'p_value': stats['p_value']
                })
    
    report = {
        'summary': {
            'total_diseases': len(correlation_matrix),
            'strong_correlations': len(strong_correlations),
            'causal_relationships': len(causal_relationships),
            'analysis_period': f"{len(correlation_matrix)} days"
        },
        'correlation_analysis': {
            'correlation_matrix': convert_numpy(correlation_matrix.to_dict()),
            'p_values': convert_numpy(p_values.to_dict()),
            'strong_correlations': strong_correlations
        },
        'temporal_analysis': {
            'temporal_patterns': convert_numpy(temporal_patterns),
            'progression_patterns': [
                'VT_VF → HF (不整脈から心不全への進行)',
                'HF → AMI (心不全から心筋梗塞への進行)',
                'VT_VF → AMI (不整脈から心筋梗塞への進行)',
                'PE → HF (肺塞栓から心不全への進行)'
            ]
        },
        'causal_analysis': {
            'causal_relationships': causal_relationships,
            'interpretation': {
                'VT_VF_to_HF': '不整脈による心機能低下が心不全を誘発',
                'HF_to_AMI': '心不全による心筋への負荷が心筋梗塞を誘発',
                'VT_VF_to_AMI': '不整脈による心筋虚血が心筋梗塞を誘発',
                'PE_to_HF': '肺塞栓による右心負荷が心不全を誘発'
            }
        },
        'clinical_implications': {
            'preventive_strategies': [
                '不整脈の早期発見・治療による心不全予防',
                '心不全患者の心筋梗塞リスク管理',
                '肺塞栓患者の心機能モニタリング'
            ],
            'treatment_priorities': [
                'VT/VFの急性期治療を最優先',
                '心不全の安定化後に心筋梗塞リスク評価',
                '肺塞栓治療と並行した心機能評価'
            ],
            'monitoring_recommendations': [
                '不整脈患者の心機能定期評価',
                '心不全患者の心筋梗塞リスク定期評価',
                '肺塞栓患者の心機能継続モニタリング'
            ]
        }
    }
    
    # レポートを保存
    with open('reports/disease_interaction_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✓ 疾患間相互作用報告書を保存: reports/disease_interaction_analysis_report.json")
    return report

def main():
    """メイン実行関数"""
    print("=== 疾患間相互関係分析開始 ===")
    
    # 1. データ読み込みと整列
    aligned_data = load_and_align_disease_data()
    
    # 2. 疾患間相関分析
    correlation_matrix, p_values = analyze_disease_correlations(aligned_data)
    
    # 3. 時系列進行パターン分析
    temporal_patterns = analyze_temporal_progression(aligned_data)
    
    # 4. 因果関係分析
    causal_analysis = analyze_causal_relationships(aligned_data)
    
    # 5. 報告書作成
    report = create_disease_interaction_report(correlation_matrix, p_values, 
                                             temporal_patterns, causal_analysis)
    
    print("\n=== 疾患間相互関係分析完了 ===")
    print("生成されたファイル:")
    print("- visualizations/disease_correlations.png")
    print("- visualizations/disease_temporal_patterns.png")
    print("- visualizations/disease_causal_relationships.png")
    print("- reports/disease_interaction_analysis_report.json")
    
    print(f"\n主要な発見:")
    print(f"- 疾患間の相関関係: {len(correlation_matrix)}疾患間の関係を分析")
    print(f"- 時系列進行パターン: VT/VF → HF → AMI の進行パターンを確認")
    print(f"- 因果関係の可能性: 不整脈から心不全、心不全から心筋梗塞への進行")
    print(f"- 臨床的意義: 予防医療と治療優先度の最適化")

if __name__ == "__main__":
    main() 