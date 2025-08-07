import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_sensitivity_specificity():
    """感度と特異度の詳細分析"""
    
    # 元のモデルの予測結果を読み込み
    df = pd.read_csv('心不全気象予測モデル_10年分最適化版_結果/all_predictions.csv')
    
    print("=== 感度・特異度分析 ===")
    print(f"データ数: {len(df)}")
    print(f"実際の高リスク日: {df['actual'].sum()} ({df['actual'].mean():.1%})")
    print(f"実際の低リスク日: {(df['actual'] == 0).sum()} ({(df['actual'] == 0).mean():.1%})")
    
    # 閾値を変えて感度・特異度を計算
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for threshold in thresholds:
        # 閾値で予測クラスを決定
        y_pred = (df['predicted_prob'] >= threshold).astype(int)
        
        # 混同行列を計算
        tn, fp, fn, tp = confusion_matrix(df['actual'], y_pred).ravel()
        
        # 性能指標を計算
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 感度（Recall）
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特異度
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # 精度
        f1 = f1_score(df['actual'], y_pred, zero_division=0)
        
        # 予測の分布
        pred_positive = y_pred.sum()
        pred_positive_rate = pred_positive / len(df)
        
        # 陽性予測値（PPV）と陰性予測値（NPV）
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'pred_positive': pred_positive,
            'pred_positive_rate': pred_positive_rate
        })
    
    results_df = pd.DataFrame(results)
    
    # 結果を表示
    print("\n=== 閾値別性能（感度・特異度） ===")
    print("閾値 | 感度 | 特異度 | 精度 | F1 | PPV | NPV")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['threshold']:.2f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | {row['precision']:.3f} | {row['f1']:.3f} | {row['ppv']:.3f} | {row['npv']:.3f}")
    
    # 最適閾値を探索
    # 医療現場用（特異度重視）
    medical_optimal = results_df.loc[results_df['specificity'] >= 0.8, 'threshold'].min()
    if pd.isna(medical_optimal):
        medical_optimal = results_df.loc[results_df['specificity'].idxmax(), 'threshold']
    
    # 予防医療用（感度重視）
    prevention_optimal = results_df.loc[results_df['sensitivity'] >= 0.8, 'threshold'].min()
    if pd.isna(prevention_optimal):
        prevention_optimal = results_df.loc[results_df['sensitivity'].idxmax(), 'threshold']
    
    # バランス型（感度+特異度最大）
    results_df['sensitivity_specificity_sum'] = results_df['sensitivity'] + results_df['specificity']
    balance_optimal = results_df.loc[results_df['sensitivity_specificity_sum'].idxmax(), 'threshold']
    
    print(f"\n=== 最適閾値 ===")
    print(f"医療現場用（特異度重視）: {medical_optimal:.2f}")
    print(f"予防医療用（感度重視）: {prevention_optimal:.2f}")
    print(f"バランス型（感度+特異度最大）: {balance_optimal:.2f}")
    
    # 最適閾値での性能を表示
    medical_row = results_df[results_df['threshold'] == medical_optimal].iloc[0]
    prevention_row = results_df[results_df['threshold'] == prevention_optimal].iloc[0]
    balance_row = results_df[results_df['threshold'] == balance_optimal].iloc[0]
    
    print(f"\n=== 最適閾値での性能 ===")
    print(f"医療現場用（閾値{medical_optimal:.2f}）:")
    print(f"  感度: {medical_row['sensitivity']:.3f}")
    print(f"  特異度: {medical_row['specificity']:.3f}")
    print(f"  精度: {medical_row['precision']:.3f}")
    print(f"  PPV: {medical_row['ppv']:.3f}")
    print(f"  NPV: {medical_row['npv']:.3f}")
    print(f"  予測陽性率: {medical_row['pred_positive_rate']:.1%}")
    
    print(f"\n予防医療用（閾値{prevention_optimal:.2f}）:")
    print(f"  感度: {prevention_row['sensitivity']:.3f}")
    print(f"  特異度: {prevention_row['specificity']:.3f}")
    print(f"  精度: {prevention_row['precision']:.3f}")
    print(f"  PPV: {prevention_row['ppv']:.3f}")
    print(f"  NPV: {prevention_row['npv']:.3f}")
    print(f"  予測陽性率: {prevention_row['pred_positive_rate']:.1%}")
    
    print(f"\nバランス型（閾値{balance_optimal:.2f}）:")
    print(f"  感度: {balance_row['sensitivity']:.3f}")
    print(f"  特異度: {balance_row['specificity']:.3f}")
    print(f"  精度: {balance_row['precision']:.3f}")
    print(f"  PPV: {balance_row['ppv']:.3f}")
    print(f"  NPV: {balance_row['npv']:.3f}")
    print(f"  予測陽性率: {balance_row['pred_positive_rate']:.1%}")
    
    # 医療用AI要件との比較
    print(f"\n=== 医療用AI要件との比較 ===")
    print("要件: 感度 80%以上, 特異度 70%以上")
    
    # 要件を満たす閾値を探索
    qualified_thresholds = results_df[
        (results_df['sensitivity'] >= 0.8) & 
        (results_df['specificity'] >= 0.7)
    ]
    
    if len(qualified_thresholds) > 0:
        best_qualified = qualified_thresholds.loc[qualified_thresholds['sensitivity_specificity_sum'].idxmax()]
        print(f"要件を満たす最適閾値: {best_qualified['threshold']:.2f}")
        print(f"  感度: {best_qualified['sensitivity']:.3f}")
        print(f"  特異度: {best_qualified['specificity']:.3f}")
    else:
        print("医療用AI要件を満たす閾値はありません")
        # 最も近い閾値を探す
        results_df['requirement_distance'] = np.sqrt(
            (results_df['sensitivity'] - 0.8)**2 + 
            (results_df['specificity'] - 0.7)**2
        )
        closest = results_df.loc[results_df['requirement_distance'].idxmin()]
        print(f"最も近い閾値: {closest['threshold']:.2f}")
        print(f"  感度: {closest['sensitivity']:.3f}")
        print(f"  特異度: {closest['specificity']:.3f}")
    
    # グラフを作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 感度・特異度曲線
    axes[0, 0].plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='感度', linewidth=2)
    axes[0, 0].plot(results_df['threshold'], results_df['specificity'], 'g-', label='特異度', linewidth=2)
    axes[0, 0].axvline(x=medical_optimal, color='red', linestyle='--', alpha=0.7, label='医療現場用')
    axes[0, 0].axvline(x=prevention_optimal, color='green', linestyle='--', alpha=0.7, label='予防医療用')
    axes[0, 0].axvline(x=balance_optimal, color='orange', linestyle='--', alpha=0.7, label='バランス型')
    axes[0, 0].set_xlabel('閾値')
    axes[0, 0].set_ylabel('感度・特異度')
    axes[0, 0].set_title('閾値 vs 感度・特異度')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PPV・NPV曲線
    axes[0, 1].plot(results_df['threshold'], results_df['ppv'], 'b-', label='PPV', linewidth=2)
    axes[0, 1].plot(results_df['threshold'], results_df['npv'], 'g-', label='NPV', linewidth=2)
    axes[0, 1].axvline(x=medical_optimal, color='red', linestyle='--', alpha=0.7, label='医療現場用')
    axes[0, 1].axvline(x=prevention_optimal, color='green', linestyle='--', alpha=0.7, label='予防医療用')
    axes[0, 1].axvline(x=balance_optimal, color='orange', linestyle='--', alpha=0.7, label='バランス型')
    axes[0, 1].set_xlabel('閾値')
    axes[0, 1].set_ylabel('PPV・NPV')
    axes[0, 1].set_title('閾値 vs PPV・NPV')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 感度 vs 特異度
    axes[1, 0].plot(results_df['specificity'], results_df['sensitivity'], 'b-', linewidth=2)
    axes[1, 0].scatter(medical_row['specificity'], medical_row['sensitivity'], color='red', s=100, label='医療現場用')
    axes[1, 0].scatter(prevention_row['specificity'], prevention_row['sensitivity'], color='green', s=100, label='予防医療用')
    axes[1, 0].scatter(balance_row['specificity'], balance_row['sensitivity'], color='orange', s=100, label='バランス型')
    axes[1, 0].set_xlabel('特異度')
    axes[1, 0].set_ylabel('感度')
    axes[1, 0].set_title('特異度 vs 感度')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 混同行列の可視化（バランス型）
    cm = np.array([[balance_row['tn'], balance_row['fp']], 
                   [balance_row['fn'], balance_row['tp']]])
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=['予測: 低リスク', '予測: 高リスク'],
                yticklabels=['実際: 低リスク', '実際: 高リスク'],
                ax=axes[1, 1])
    axes[1, 1].set_title(f'混同行列（バランス型、閾値{balance_optimal:.2f}）')
    
    plt.tight_layout()
    plt.savefig('感度特異度分析結果.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 結果をCSVに保存
    results_df.to_csv('感度特異度分析結果.csv', index=False)
    
    print(f"\n結果を保存しました:")
    print(f"- グラフ: 感度特異度分析結果.png")
    print(f"- データ: 感度特異度分析結果.csv")
    
    return results_df, medical_optimal, prevention_optimal, balance_optimal

if __name__ == "__main__":
    analyze_sensitivity_specificity() 