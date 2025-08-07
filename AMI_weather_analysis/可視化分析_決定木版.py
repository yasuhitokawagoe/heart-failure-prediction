import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def get_saved_model_features():
    """保存されたモデルの特徴量セットを取得"""
    feature_importance_df = pd.read_csv('保存モデル/結果/feature_importance.csv')
    saved_features = feature_importance_df['feature'].tolist()
    return saved_features

def encode_simplified_weather_conditions(df):
    """天気概況を6分類に簡略化して数値化"""
    weather_mapping = {
        '晴れ': '晴れ系', '快晴': '晴れ系', '曇り': '曇り系', '薄曇': '曇り系',
        '小雨': '小雨', '大雨': '大雨', '雷雨': '雷雨', '雪': '雪'
    }
    df['weather_simplified'] = df['天気分類(統合)'].map(weather_mapping)
    df['weather_simplified'] = df['weather_simplified'].fillna('曇り系')
    le = LabelEncoder()
    df['weather_simplified_encoded'] = le.fit_transform(df['weather_simplified'])
    return df, le

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    import jpholiday
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end.astype(int)
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def create_weather_interaction_features(df):
    """気象要素の相互作用特徴量を作成"""
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_pressure'] = df['avg_temp'] * df['vapor_pressure']
    df['humidity_pressure'] = df['avg_humidity'] * df['vapor_pressure']
    
    df['temp_change'] = df['avg_temp'].diff()
    df['humidity_change'] = df['avg_humidity'].diff()
    df['pressure_change'] = df['vapor_pressure'].diff()
    
    df['temp_change_humidity_change'] = df['temp_change'] * df['humidity_change']
    df['temp_change_pressure_change'] = df['temp_change'] * df['pressure_change']
    df['humidity_change_pressure_change'] = df['humidity_change'] * df['pressure_change']
    
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    return df

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成"""
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    
    for col in weather_cols:
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].rolling(window=window, min_periods=window).std()
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
    
    return df

def create_seasonal_weighted_features(df):
    """季節性を考慮した重み付け特徴量を作成"""
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    weather_cols = ['avg_temp', 'avg_humidity', 'vapor_pressure']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def create_additional_features(df):
    """保存されたモデルに必要な追加特徴量を作成"""
    # 月次変化率
    df['avg_temp_monthly_change_rate'] = df['avg_temp'].pct_change(30)
    df['avg_humidity_monthly_change_rate'] = df['avg_humidity'].pct_change(30)
    
    # 週次変化率
    df['vapor_pressure_weekly_change_rate'] = df['vapor_pressure'].pct_change(7)
    df['avg_humidity_weekly_change_rate'] = df['avg_humidity'].pct_change(7)
    
    # 季節性偏差
    df['temp_seasonal_deviation_7d'] = df['avg_temp'] - df['avg_temp'].rolling(7).mean()
    df['temp_seasonal_deviation_14d'] = df['avg_temp'] - df['avg_temp'].rolling(14).mean()
    df['seasonal_temp_deviation'] = df['avg_temp'] - df.groupby('month')['avg_temp'].transform('mean')
    
    # トレンド強度
    df['avg_temp_trend_strength'] = df['avg_temp'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['avg_humidity_trend_strength'] = df['avg_humidity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    df['vapor_pressure_trend_strength'] = df['vapor_pressure'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    # トレンドスロープ
    df['avg_humidity_trend_slope'] = df['avg_humidity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    # 3日変化
    df['temp_change_3d'] = df['avg_temp'].diff(3)
    
    # 異常スコア
    df['avg_temp_anomaly_score'] = (df['avg_temp'] - df['avg_temp'].rolling(30).mean()) / df['avg_temp'].rolling(30).std()
    df['avg_humidity_anomaly_score'] = (df['avg_humidity'] - df['avg_humidity'].rolling(30).mean()) / df['avg_humidity'].rolling(30).std()
    
    # 予測可能性
    df['avg_temp_predictability'] = df['avg_temp'].rolling(7).std() / df['avg_temp'].rolling(30).std()
    
    # 加速度
    df['avg_temp_acceleration'] = df['avg_temp'].diff().diff()
    df['avg_temp_acceleration_2nd'] = df['avg_temp'].diff().diff().diff()
    
    # 気圧変化と温度変化の相互作用
    df['pressure_change_temp_change'] = df['pressure_change'] * df['temp_change']
    
    return df

def load_model_and_data():
    """保存されたモデルとデータを読み込み"""
    try:
        model = joblib.load('保存モデル/AMI予測モデル_天気概況6分類版_XGBoost.pkl')
        
        df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
        df['hospitalization_date'] = pd.to_datetime(df['date'])
        df['hospitalization_count'] = df['people']
        
        return model, df
    except Exception as e:
        print(f"エラー: {e}")
        return None, None

def prepare_data_with_saved_features(df, saved_features):
    """保存されたモデルの特徴量セットでデータを準備"""
    # 日付でソート
    df = df.sort_values('hospitalization_date')
    
    # 天気概況を6分類に簡略化
    df, weather_encoder = encode_simplified_weather_conditions(df)
    
    # ターゲット変数の作成
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    # 特徴量エンジニアリング
    df = create_date_features(df)
    df = create_weather_interaction_features(df)
    df = create_time_series_features(df)
    df = create_advanced_weather_timeseries_features(df)
    df = create_seasonal_weighted_features(df)
    df = create_additional_features(df)
    
    # 保存されたモデルの特徴量セットを使用
    available_features = [f for f in saved_features if f in df.columns]
    
    # データを準備
    X = df[available_features].values
    y = df['target'].values
    
    # 欠損値の処理
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, available_features, df

def create_decision_tree_model(X_scaled, y, feature_names):
    """決定木モデルを作成"""
    print("=== 決定木モデルを作成 ===")
    
    # 決定木モデルを訓練
    dt_model = DecisionTreeClassifier(
        max_depth=5,  # 深さを制限して解釈可能性を向上
        min_samples_split=50,
        min_samples_leaf=25,
        random_state=42
    )
    
    dt_model.fit(X_scaled, y)
    
    return dt_model

def visualize_decision_tree(dt_model, feature_names):
    """決定木を可視化"""
    print("=== 決定木を可視化 ===")
    
    plt.figure(figsize=(20, 12))
    
    # 決定木を描画
    plot_tree(dt_model, 
              feature_names=feature_names,
              class_names=['低リスク', '高リスク'],
              filled=True,
              rounded=True,
              fontsize=8)
    
    plt.title('決定木によるAMIリスク分類', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('決定木分析結果/decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # テキスト形式でも出力
    tree_text = export_text(dt_model, feature_names=feature_names)
    with open('決定木分析結果/decision_tree_text.txt', 'w', encoding='utf-8') as f:
        f.write(tree_text)
    
    print("決定木の可視化が完了しました")

def create_decision_path_analysis(dt_model, X_scaled, feature_names, df):
    """決定パス分析"""
    print("=== 決定パス分析 ===")
    predictions = dt_model.predict(X_scaled)
    high_risk_indices = np.where(predictions == 1)[0][:5]
    low_risk_indices = np.where(predictions == 0)[0][:5]
    sample_indices = np.concatenate([high_risk_indices, low_risk_indices])

    for i, idx in enumerate(sample_indices):
        print(f"\n=== サンプル {i+1} (インデックス: {idx}) ===")
        print(f"予測: {'高リスク' if predictions[idx] == 1 else '低リスク'}")
        print(f"実際の入院数: {df.iloc[idx]['hospitalization_count']}")
        print(f"日付: {df.iloc[idx]['hospitalization_date']}")

        decision_path = dt_model.decision_path(X_scaled[idx:idx+1])
        node_indices = decision_path.indices[decision_path.indptr[0]:decision_path.indptr[1]]

        print("決定パス:")
        for j, node_idx in enumerate(node_indices):
            if dt_model.tree_.feature[node_idx] != -2:  # Check if it's a split node
                feature_idx = dt_model.tree_.feature[node_idx]
                feature_name = feature_names[feature_idx]
                feature_value = X_scaled[idx, feature_idx]
                threshold = dt_model.tree_.threshold[node_idx]
                
                # Determine the decision rule
                if X_scaled[idx, feature_idx] <= threshold:
                    rule = f"{feature_name} <= {threshold:.3f}"
                else:
                    rule = f"{feature_name} > {threshold:.3f}"

                feature_japanese = {
                    'is_holiday': '祝日・週末',
                    'vapor_pressure_std_7d': '7日間気圧変動',
                    'avg_humidity_weekly_change_rate': '湿度週次変化率',
                    'avg_temp_change_rate': '気温変化率',
                    'pressure_change_temp_change': '気圧・気温変化相互作用',
                    'avg_temp_trend_strength': '気温トレンド強度',
                    'avg_temp_monthly_change_rate': '気温月次変化率',
                    'avg_humidity_monthly_change_rate': '湿度月次変化率',
                    'seasonal_temp_deviation': '季節性気温偏差',
                    'vapor_pressure_weekly_change_rate': '気圧週次変化率',
                    'sunshine_hours': '日照時間',
                    'avg_temp_acceleration': '気温加速度',
                    'temp_change_humidity_change': '気温・湿度変化相互作用',
                    'avg_humidity_std_3d': '3日間湿度標準偏差'
                }
                japanese_name = feature_japanese.get(feature_name, feature_name)
                print(f"  {j+1}. {japanese_name} ({rule}): {feature_value:.3f}")

def create_feature_importance_comparison(dt_model, rf_model, feature_names):
    """特徴量重要度の比較"""
    print("=== 特徴量重要度の比較 ===")
    
    # 決定木の特徴量重要度
    dt_importance = dt_model.feature_importances_
    
    # ランダムフォレストの特徴量重要度
    rf_importance = rf_model.feature_importances_
    
    # 比較データフレームを作成
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'decision_tree': dt_importance,
        'random_forest': rf_importance
    })
    
    # 上位10個の特徴量を選択
    top_features = importance_df.nlargest(10, 'random_forest')
    
    # 可視化
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(top_features))
    width = 0.35
    
    plt.bar(x - width/2, top_features['decision_tree'], width, 
            label='決定木', alpha=0.7, color='skyblue')
    plt.bar(x + width/2, top_features['random_forest'], width, 
            label='ランダムフォレスト', alpha=0.7, color='lightcoral')
    
    plt.xlabel('特徴量', fontsize=12)
    plt.ylabel('重要度', fontsize=12)
    plt.title('特徴量重要度の比較', fontsize=14, fontweight='bold')
    plt.xticks(x, top_features['feature'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('決定木分析結果/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("特徴量重要度比較:")
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}:")
        print(f"    決定木: {row['decision_tree']:.4f}")
        print(f"    ランダムフォレスト: {row['random_forest']:.4f}")

def create_random_forest_model(X_scaled, y, feature_names):
    """ランダムフォレストモデルを作成"""
    print("=== ランダムフォレストモデルを作成 ===")
    
    # ランダムフォレストモデルを訓練
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    rf_model.fit(X_scaled, y)
    
    return rf_model

def create_decision_boundary_analysis(dt_model, X_scaled, feature_names):
    """決定境界分析"""
    print("=== 決定境界分析 ===")
    
    # 重要な2つの特徴量を選択
    important_features = ['is_holiday', 'vapor_pressure_std_7d']
    
    if important_features[0] in feature_names and important_features[1] in feature_names:
        idx1 = feature_names.index(important_features[0])
        idx2 = feature_names.index(important_features[1])
        
        # データを2次元に投影
        X_2d = X_scaled[:, [idx1, idx2]]
        
        # 2次元データで決定木を再訓練
        dt_2d = DecisionTreeClassifier(max_depth=5, random_state=42)
        y = dt_model.predict(X_scaled)  # 元のモデルの予測を教師データとして使用
        dt_2d.fit(X_2d, y)
        
        # 決定境界を可視化
        plt.figure(figsize=(12, 8))
        
        # メッシュグリッドを作成
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 予測
        Z = dt_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 決定境界を描画
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # データポイントを描画
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                             c=y, 
                             cmap='RdYlBu', alpha=0.6, s=20)
        
        plt.xlabel(f'{important_features[0]} (標準化値)', fontsize=12)
        plt.ylabel(f'{important_features[1]} (標準化値)', fontsize=12)
        plt.title('決定境界分析', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='予測クラス')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('決定木分析結果/decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"決定境界分析を完了しました: {important_features[0]} vs {important_features[1]}")
    else:
        print("必要な特徴量が見つかりません。決定境界分析をスキップします。")

def create_detailed_report():
    """詳細な分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    with open('決定木分析結果/決定木分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# 決定木分析レポート\n\n')
        
        f.write('## 概要\n')
        f.write('- 決定木による解釈可能な分類モデル\n')
        f.write('- 決定パスの詳細分析\n')
        f.write('- 特徴量重要度の比較\n')
        f.write('- 決定境界の可視化\n\n')
        
        f.write('## 主要な発見\n\n')
        
        f.write('### 1. 決定木の構造\n')
        f.write('- 階層的な意思決定プロセスの可視化\n')
        f.write('- 各分岐点での特徴量の閾値\n')
        f.write('- リスク分類の論理的根拠\n\n')
        
        f.write('### 2. 決定パス分析\n')
        f.write('- 個別サンプルの分類過程を追跡\n')
        f.write('- 重要な特徴量の順序\n')
        f.write('- 医学的解釈に基づく検証\n\n')
        
        f.write('### 3. 特徴量重要度の比較\n')
        f.write('- 決定木とランダムフォレストの比較\n')
        f.write('- 一貫性のある重要特徴量の特定\n')
        f.write('- モデル間の安定性評価\n\n')
        
        f.write('### 4. 決定境界分析\n')
        f.write('- 2次元空間での分類境界\n')
        f.write('- 特徴量間の相互作用\n')
        f.write('- 分類の不確実性領域\n\n')
        
        f.write('## 医学的解釈\n')
        f.write('- 階層的なリスク評価プロセス\n')
        f.write('- 臨床意思決定の透明性向上\n')
        f.write('- 予防医療における段階的アプローチ\n')
        f.write('- 個別化されたリスク評価の実現\n')
        f.write('- 医療従事者への説明可能性\n')
    
    print("詳細レポートを保存しました: 決定木分析結果/決定木分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        import os
        os.makedirs('決定木分析結果', exist_ok=True)
        
        print("保存されたモデルの特徴量セットを取得中...")
        saved_features = get_saved_model_features()
        
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        print("データを準備中...")
        X_scaled, y, feature_names, df = prepare_data_with_saved_features(df, saved_features)
        
        print("決定木分析を実行中...")
        
        # 1. 決定木モデルの作成
        dt_model = create_decision_tree_model(X_scaled, y, feature_names)
        
        # 2. 決定木の可視化
        visualize_decision_tree(dt_model, feature_names)
        
        # 3. 決定パス分析
        create_decision_path_analysis(dt_model, X_scaled, feature_names, df)
        
        # 4. ランダムフォレストモデルの作成
        rf_model = create_random_forest_model(X_scaled, y, feature_names)
        
        # 5. 特徴量重要度の比較
        create_feature_importance_comparison(dt_model, rf_model, feature_names)
        
        # 6. 決定境界分析
        create_decision_boundary_analysis(dt_model, X_scaled, feature_names)
        
        # 7. 詳細レポート
        create_detailed_report()
        
        print("\n=== 決定木分析完了 ===")
        print("結果は '決定木分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 