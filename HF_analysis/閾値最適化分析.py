import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_threshold_optimization():
    """元のモデルの予測結果を使って閾値最適化を分析"""
    
    # 元のモデルの予測結果を読み込み
    df = pd.read_csv('心不全気象予測モデル_10年分最適化版_結果/all_predictions.csv')
    
    print("=== 閾値最適化分析 ===")
    print(f"データ数: {len(df)}")
    print(f"実際の高リスク日: {df['actual'].sum()} ({df['actual'].mean():.1%})")
    print(f"実際の低リスク日: {(df['actual'] == 0).sum()} ({(df['actual'] == 0).mean():.1%})")
    
    # 閾値を変えてPrecision/Recallを計算
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        # 閾値で予測クラスを決定
        y_pred = (df['predicted_prob'] >= threshold).astype(int)
        
        # 性能指標を計算
        precision = precision_score(df['actual'], y_pred, zero_division=0)
        recall = recall_score(df['actual'], y_pred, zero_division=0)
        f1 = f1_score(df['actual'], y_pred, zero_division=0)
        
        # 予測の分布
        pred_positive = y_pred.sum()
        pred_positive_rate = pred_positive / len(df)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_positive': pred_positive,
            'pred_positive_rate': pred_positive_rate
        })
    
    results_df = pd.DataFrame(results)
    
    # 結果を表示
    print("\n=== 閾値別性能 ===")
    print("閾値 | Precision | Recall | F1 | 予測陽性率")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['pred_positive_rate']:.1%}")
    
    # 最適閾値を探索
    # 医療現場用（Precision重視）
    medical_optimal = results_df.loc[results_df['precision'] >= 0.7, 'threshold'].min()
    if pd.isna(medical_optimal):
        medical_optimal = results_df.loc[results_df['precision'].idxmax(), 'threshold']
    
    # 予防医療用（Recall重視）
    prevention_optimal = results_df.loc[results_df['recall'] >= 0.6, 'threshold'].min()
    if pd.isna(prevention_optimal):
        prevention_optimal = results_df.loc[results_df['recall'].idxmax(), 'threshold']
    
    # バランス型（F1最大）
    balance_optimal = results_df.loc[results_df['f1'].idxmax(), 'threshold']
    
    print(f"\n=== 最適閾値 ===")
    print(f"医療現場用（Precision重視）: {medical_optimal:.2f}")
    print(f"予防医療用（Recall重視）: {prevention_optimal:.2f}")
    print(f"バランス型（F1最大）: {balance_optimal:.2f}")
    
    # 最適閾値での性能を表示
    medical_row = results_df[results_df['threshold'] == medical_optimal].iloc[0]
    prevention_row = results_df[results_df['threshold'] == prevention_optimal].iloc[0]
    balance_row = results_df[results_df['threshold'] == balance_optimal].iloc[0]
    
    print(f"\n=== 最適閾値での性能 ===")
    print(f"医療現場用（閾値{medical_optimal:.2f}）:")
    print(f"  Precision: {medical_row['precision']:.3f}")
    print(f"  Recall: {medical_row['recall']:.3f}")
    print(f"  F1: {medical_row['f1']:.3f}")
    print(f"  予測陽性率: {medical_row['pred_positive_rate']:.1%}")
    
    print(f"\n予防医療用（閾値{prevention_optimal:.2f}）:")
    print(f"  Precision: {prevention_row['precision']:.3f}")
    print(f"  Recall: {prevention_row['recall']:.3f}")
    print(f"  F1: {prevention_row['f1']:.3f}")
    print(f"  予測陽性率: {prevention_row['pred_positive_rate']:.1%}")
    
    print(f"\nバランス型（閾値{balance_optimal:.2f}）:")
    print(f"  Precision: {balance_row['precision']:.3f}")
    print(f"  Recall: {balance_row['recall']:.3f}")
    print(f"  F1: {balance_row['f1']:.3f}")
    print(f"  予測陽性率: {balance_row['pred_positive_rate']:.1%}")
    
    # グラフを作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision-Recall曲線
    axes[0, 0].plot(results_df['recall'], results_df['precision'], 'b-', linewidth=2)
    axes[0, 0].scatter(medical_row['recall'], medical_row['precision'], color='red', s=100, label='医療現場用')
    axes[0, 0].scatter(prevention_row['recall'], prevention_row['precision'], color='green', s=100, label='予防医療用')
    axes[0, 0].scatter(balance_row['recall'], balance_row['precision'], color='orange', s=100, label='バランス型')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall曲線')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 閾値 vs 性能指標
    axes[0, 1].plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
    axes[0, 1].plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
    axes[0, 1].plot(results_df['threshold'], results_df['f1'], 'r-', label='F1', linewidth=2)
    axes[0, 1].axvline(x=medical_optimal, color='red', linestyle='--', alpha=0.7, label='医療現場用')
    axes[0, 1].axvline(x=prevention_optimal, color='green', linestyle='--', alpha=0.7, label='予防医療用')
    axes[0, 1].axvline(x=balance_optimal, color='orange', linestyle='--', alpha=0.7, label='バランス型')
    axes[0, 1].set_xlabel('閾値')
    axes[0, 1].set_ylabel('性能指標')
    axes[0, 1].set_title('閾値 vs 性能指標')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 予測確率の分布
    axes[1, 0].hist(df[df['actual'] == 0]['predicted_prob'], bins=30, alpha=0.7, label='低リスク日', color='blue')
    axes[1, 0].hist(df[df['actual'] == 1]['predicted_prob'], bins=30, alpha=0.7, label='高リスク日', color='red')
    axes[1, 0].axvline(x=medical_optimal, color='red', linestyle='--', alpha=0.7, label='医療現場用閾値')
    axes[1, 0].axvline(x=prevention_optimal, color='green', linestyle='--', alpha=0.7, label='予防医療用閾値')
    axes[1, 0].axvline(x=balance_optimal, color='orange', linestyle='--', alpha=0.7, label='バランス型閾値')
    axes[1, 0].set_xlabel('予測確率')
    axes[1, 0].set_ylabel('頻度')
    axes[1, 0].set_title('予測確率の分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 予測陽性率 vs 閾値
    axes[1, 1].plot(results_df['threshold'], results_df['pred_positive_rate'], 'purple', linewidth=2)
    axes[1, 1].axvline(x=medical_optimal, color='red', linestyle='--', alpha=0.7, label='医療現場用')
    axes[1, 1].axvline(x=prevention_optimal, color='green', linestyle='--', alpha=0.7, label='予防医療用')
    axes[1, 1].axvline(x=balance_optimal, color='orange', linestyle='--', alpha=0.7, label='バランス型')
    axes[1, 1].set_xlabel('閾値')
    axes[1, 1].set_ylabel('予測陽性率')
    axes[1, 1].set_title('閾値 vs 予測陽性率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('閾値最適化分析結果.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果をCSVに保存
    results_df.to_csv('閾値最適化結果.csv', index=False)
    
    print(f"\n結果を保存しました:")
    print(f"- グラフ: 閾値最適化分析結果.png")
    print(f"- データ: 閾値最適化結果.csv")
    
    return results_df, medical_optimal, prevention_optimal, balance_optimal

if __name__ == "__main__":
    analyze_threshold_optimization() 