import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_extreme_weather_flags():
    """異常気象フラグの重要度を確認"""
    print("=== 異常気象フラグの重要度確認 ===")
    
    # モデルの読み込み
    model = joblib.load('saved_models/xgb_model_latest.pkl')
    print(f"✓ モデル読み込み完了 (特徴量数: {model.n_features_in_})")
    
    # 特徴量重要度の取得
    feature_importance = np.abs(model.feature_importances_)
    print(f"✓ 特徴量重要度取得完了 (形状: {feature_importance.shape})")
    
    # 特徴量名の生成（実際の名前は不明なので番号で）
    feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    
    # 重要度のDataFrame作成
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== 全特徴量の重要度（上位20位） ===")
    for i, row in enumerate(importance_df.head(20).itertuples(), 1):
        print(f"{i:2d}. {row.feature}: {row.importance:.4f}")
    
    # 異常気象フラグの特定（実際の名前は不明なので、特徴量の重要度パターンから推測）
    print(f"\n=== 異常気象フラグの推測 ===")
    print("注: 実際の特徴量名は保存されていないため、重要度パターンから推測")
    
    # 重要度が低い特徴量（異常気象フラグの可能性）
    low_importance_features = importance_df.tail(20)
    print(f"\n重要度の低い特徴量（異常気象フラグの可能性）:")
    for i, row in enumerate(low_importance_features.itertuples(), 1):
        print(f"{i:2d}. {row.feature}: {row.importance:.4f}")
    
    # 統計情報
    print(f"\n=== 統計情報 ===")
    print(f"全特徴量数: {len(importance_df)}")
    print(f"平均重要度: {importance_df['importance'].mean():.4f}")
    print(f"重要度の標準偏差: {importance_df['importance'].std():.4f}")
    print(f"重要度の最小値: {importance_df['importance'].min():.4f}")
    print(f"重要度の最大値: {importance_df['importance'].max():.4f}")
    
    # 重要度の分布
    print(f"\n=== 重要度の分布 ===")
    print(f"上位10%: {importance_df['importance'].quantile(0.9):.4f}")
    print(f"上位25%: {importance_df['importance'].quantile(0.75):.4f}")
    print(f"中央値: {importance_df['importance'].quantile(0.5):.4f}")
    print(f"下位25%: {importance_df['importance'].quantile(0.25):.4f}")
    print(f"下位10%: {importance_df['importance'].quantile(0.1):.4f}")

if __name__ == "__main__":
    check_extreme_weather_flags() 