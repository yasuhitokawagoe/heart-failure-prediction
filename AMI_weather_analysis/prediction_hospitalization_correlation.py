# Prediction vs Hospitalization Correlation Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_prediction_hospitalization_correlation():
    """Create correlation plots between prediction probabilities and hospitalization counts"""
    
    print("Creating prediction vs hospitalization correlation plots...")
    
    # Sample data (replace with actual data)
    sample_data = {
        'hospitalization_date': [
            '2023-08-15', '2023-03-20', '2023-12-10', '2023-07-25', '2023-01-15',
            '2023-06-30', '2023-09-15', '2023-11-05', '2023-04-10', '2023-02-28'
        ],
        'hospitalization_count': [3, 0, 2, 4, 1, 2, 1, 3, 0, 1],
        'prediction_probability': [0.892, 0.156, 0.734, 0.945, 0.623, 0.456, 0.234, 0.567, 0.189, 0.345]
    }
    
    df = pd.DataFrame(sample_data)
    df['hospitalization_date'] = pd.to_datetime(df['hospitalization_date'])
    
    # Create correlation plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Prediction vs Hospitalization
    ax1.scatter(df['prediction_probability'], df['hospitalization_count'], 
                alpha=0.7, s=100, color='blue')
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Hospitalization Count', fontsize=12)
    ax1.set_title('Prediction Probability vs Hospitalization Count', fontsize=14, fontweight='bold')
    
    # Add correlation coefficient
    corr = np.corrcoef(df['prediction_probability'], df['hospitalization_count'])[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Box plot by prediction quartiles
    df['prediction_quartile'] = pd.qcut(df['prediction_probability'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    sns.boxplot(data=df, x='prediction_quartile', y='hospitalization_count', ax=ax2)
    ax2.set_xlabel('Prediction Probability Quartile', fontsize=12)
    ax2.set_ylabel('Hospitalization Count', fontsize=12)
    ax2.set_title('Hospitalization Count by Prediction Quartile', fontsize=14, fontweight='bold')
    
    # 3. Time series plot
    ax3.plot(df['hospitalization_date'], df['prediction_probability'], 
             marker='o', linewidth=2, label='Prediction Probability', color='red')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Prediction Probability', fontsize=12)
    ax3.set_title('Prediction Probability Over Time', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Dual axis plot
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(df['hospitalization_date'], df['hospitalization_count'], 
                     marker='s', linewidth=2, label='Hospitalization Count', color='blue')
    line2 = ax4_twin.plot(df['hospitalization_date'], df['prediction_probability'], 
                          marker='o', linewidth=2, label='Prediction Probability', color='red')
    
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Hospitalization Count', fontsize=12, color='blue')
    ax4_twin.set_ylabel('Prediction Probability', fontsize=12, color='red')
    ax4.set_title('Prediction vs Actual Hospitalization', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('prediction_hospitalization_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Correlation plots saved as 'prediction_hospitalization_correlation.png'")
    
    # Print correlation statistics
    print(f"\nCorrelation Statistics:")
    print(f"Pearson correlation: {corr:.3f}")
    print(f"Spearman correlation: {df['prediction_probability'].corr(df['hospitalization_count'], method='spearman'):.3f}")
    
    return df

if __name__ == "__main__":
    df = create_prediction_hospitalization_correlation() 