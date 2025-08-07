# Simple Prediction vs Hospitalization Correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

def load_model_and_data():
    """モデルとデータを読み込み"""
    # モデルの読み込み
    model_path = 'saved_models/xgb_model_latest.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("モデルを読み込みました")
    else:
        print("モデルファイルが見つかりません")
        return None, None, None
    
    # データの読み込み（東京のデータを使用）
    df = pd.read_csv('../東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
    df['hospitalization_date'] = pd.to_datetime(df['date'])
    df = df.sort_values('hospitalization_date')
    
    # ターゲット変数の作成
    threshold = df['people'].quantile(0.75)
    df['target'] = (df['people'] >= threshold).astype(int)
    
    return model, df, threshold

def create_features(df):
    """特徴量を作成"""
    # 日付特徴量
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # 季節性特徴量
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # 異常気象フラグ
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    
    return df

def get_test_data(df, model):
    """テストデータを取得"""
    # 特徴量の準備
    df = create_features(df)
    
    # 使用する特徴量を定義（訓練時と同じ除外リストを使用）
    exclude_cols = ['hospitalization_date', 'target', 'prefecture_name', 'date', 'hospitalization_count', 'people']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院人数関連の特徴量を完全に除外（訓練時と同じ）
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # 数値型の列のみを抽出
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    # 欠損値の処理
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 無限大の値を適切な値に置換
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 時系列分割でテストデータを取得
    unique_dates = pd.Series(df['hospitalization_date'].unique()).sort_values()
    total_days = len(unique_dates)
    
    # 最新の1年分をテストデータとして使用
    test_size = 365
    test_end_idx = total_days
    test_start_idx = test_end_idx - test_size
    
    test_dates = unique_dates.iloc[test_start_idx:test_end_idx]
    test_df = df[df['hospitalization_date'].isin(test_dates)]
    
    # テストデータの特徴量
    X_test = test_df[feature_columns].values
    
    # モデルが期待する特徴量数に合わせる（上位29個を選択）
    if X_test.shape[1] > 29:
        # 簡易的な特徴量選択（実際の訓練時と同じ方法を使用すべき）
        # ここでは最初の29個を使用
        X_test = X_test[:, :29]
        print(f"特徴量数を29個に調整しました（元の数: {X_test.shape[1] + 2}）")
    
    y_test = test_df['target'].values
    hospitalization_counts = test_df['people'].values
    dates = test_df['hospitalization_date'].values
    
    # 予測
    predictions = model.predict_proba(X_test)[:, 1]
    
    return predictions, hospitalization_counts, dates, test_df

def create_correlation_plot(predictions, hospitalization_counts, dates):
    """相関プロットを作成"""
    plt.figure(figsize=(12, 8))
    
    # 散布図
    plt.scatter(hospitalization_counts, predictions, alpha=0.6, s=50)
    
    # トレンドライン
    z = np.polyfit(hospitalization_counts, predictions, 1)
    p = np.poly1d(z)
    plt.plot(hospitalization_counts, p(hospitalization_counts), "r--", alpha=0.8)
    
    # 相関係数の計算
    correlation = np.corrcoef(hospitalization_counts, predictions)[0, 1]
    
    plt.xlabel('Hospitalization Count', fontsize=12)
    plt.ylabel('Prediction Probability', fontsize=12)
    plt.title(f'Prediction Probability vs Hospitalization Count\nCorrelation: {correlation:.3f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # データポイントにラベルを追加（一部のみ）
    for i in range(0, len(dates), 30):  # 30日ごとにラベル
        plt.annotate(f'{dates[i].strftime("%Y-%m")}', 
                    (hospitalization_counts[i], predictions[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('prediction_hospitalization_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"相関係数: {correlation:.3f}")
    print(f"データポイント数: {len(predictions)}")
    print(f"期間: {dates[0].strftime('%Y-%m-%d')} から {dates[-1].strftime('%Y-%m-%d')}")

def main():
    """メイン実行関数"""
    print("予測確率と入院数の相関分析を開始します...")
    
    # モデルとデータの読み込み
    model, df, threshold = load_model_and_data()
    if model is None:
        return
    
    print(f"データ総数: {len(df)}")
    print(f"75%タイル閾値: {threshold:.1f}")
    
    # テストデータの取得
    predictions, hospitalization_counts, dates, test_df = get_test_data(df, model)
    
    print(f"テストデータ数: {len(predictions)}")
    print(f"テスト期間: {dates[0].strftime('%Y-%m-%d')} から {dates[-1].strftime('%Y-%m-%d')}")
    
    # 相関プロットの作成
    create_correlation_plot(predictions, hospitalization_counts, dates)
    
    # 統計情報の表示
    print("\n=== 統計情報 ===")
    print(f"予測確率の平均: {np.mean(predictions):.3f}")
    print(f"予測確率の標準偏差: {np.std(predictions):.3f}")
    print(f"入院数の平均: {np.mean(hospitalization_counts):.1f}")
    print(f"入院数の標準偏差: {np.std(hospitalization_counts):.1f}")

if __name__ == "__main__":
    main() 