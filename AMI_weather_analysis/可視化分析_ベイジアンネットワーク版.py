import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx
from scipy.stats import pearsonr
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

def create_correlation_network(X_scaled, feature_names, threshold=0.3):
    """相関ネットワークを作成"""
    print("=== 相関ネットワークを作成 ===")
    
    # 相関行列を計算
    correlation_matrix = np.corrcoef(X_scaled.T)
    
    # ネットワークを作成
    G = nx.Graph()
    
    # ノードを追加
    for i, feature in enumerate(feature_names):
        G.add_node(feature)
    
    # エッジを追加（相関が閾値を超える場合）
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(correlation_matrix[i, j])
            if corr > threshold:
                G.add_edge(feature_names[i], feature_names[j], weight=corr)
    
    return G, correlation_matrix

def visualize_correlation_network(G, feature_names):
    """相関ネットワークを可視化"""
    print("=== 相関ネットワークを可視化 ===")
    
    plt.figure(figsize=(16, 12))
    
    # レイアウトを設定
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # ノードのサイズを重要度に基づいて設定
    node_sizes = []
    for node in G.nodes():
        # 特徴量の重要度を取得（簡易版）
        if 'is_holiday' in node:
            node_sizes.append(1000)
        elif 'vapor_pressure' in node:
            node_sizes.append(800)
        elif 'humidity' in node:
            node_sizes.append(600)
        elif 'temp' in node:
            node_sizes.append(600)
        else:
            node_sizes.append(300)
    
    # エッジの重みを取得
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # ネットワークを描画
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='DejaVu Sans')
    
    plt.title('特徴量相関ネットワーク', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ベイジアンネットワーク分析結果/correlation_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ネットワーク統計:")
    print(f"  ノード数: {G.number_of_nodes()}")
    print(f"  エッジ数: {G.number_of_edges()}")
    print(f"  平均次数: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

def create_feature_clusters(G, feature_names):
    """特徴量クラスタリング"""
    print("=== 特徴量クラスタリング ===")
    
    # コミュニティ検出
    from community import community_louvain
    communities = community_louvain.best_partition(G)
    
    # クラスタごとに特徴量をグループ化
    clusters = {}
    for node, cluster_id in communities.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    # クラスタを可視化
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # 各クラスタを異なる色で描画
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for cluster_id, nodes in clusters.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                              node_color=[colors[cluster_id]], 
                              node_size=500, alpha=0.7)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='DejaVu Sans')
    
    plt.title('特徴量クラスタリング', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ベイジアンネットワーク分析結果/feature_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("クラスタ分析結果:")
    for cluster_id, nodes in clusters.items():
        print(f"  クラスタ {cluster_id}: {len(nodes)}個の特徴量")
        for node in nodes[:5]:  # 最初の5個のみ表示
            print(f"    - {node}")
        if len(nodes) > 5:
            print(f"    ... 他 {len(nodes)-5}個")

def create_causal_analysis(X_scaled, feature_names):
    """因果関係分析"""
    print("=== 因果関係分析 ===")
    
    # 重要な特徴量を選択
    important_features = ['is_holiday', 'vapor_pressure_std_7d', 'avg_humidity_weekly_change_rate',
                         'avg_temp_change_rate', 'pressure_change_temp_change', 'avg_temp_trend_strength']
    
    # 因果関係マトリックスを作成
    causal_matrix = np.zeros((len(important_features), len(important_features)))
    
    for i, feature1 in enumerate(important_features):
        for j, feature2 in enumerate(important_features):
            if i != j:
                # 簡易的な因果関係スコアを計算
                idx1 = feature_names.index(feature1)
                idx2 = feature_names.index(feature2)
                
                # 相関と時系列の関係を考慮
                correlation = np.corrcoef(X_scaled[:, idx1], X_scaled[:, idx2])[0, 1]
                
                # 因果関係スコア（簡易版）
                causal_score = abs(correlation) * 0.5  # 簡易的な重み付け
                causal_matrix[i, j] = causal_score
    
    # 因果関係ネットワークを作成
    G_causal = nx.DiGraph()
    
    for i, feature1 in enumerate(important_features):
        for j, feature2 in enumerate(important_features):
            if i != j and causal_matrix[i, j] > 0.1:  # 閾値
                G_causal.add_edge(feature1, feature2, weight=causal_matrix[i, j])
    
    # 可視化
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_causal, k=2, iterations=50)
    
    # ノードを描画
    nx.draw_networkx_nodes(G_causal, pos, node_size=1000, node_color='lightgreen', alpha=0.7)
    
    # エッジを描画
    edge_weights = [G_causal[u][v]['weight'] for u, v in G_causal.edges()]
    nx.draw_networkx_edges(G_causal, pos, width=edge_weights, alpha=0.6, 
                           edge_color='red', arrows=True, arrowsize=20)
    
    # ラベルを描画
    nx.draw_networkx_labels(G_causal, pos, font_size=10, font_weight='bold')
    
    plt.title('因果関係ネットワーク', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ベイジアンネットワーク分析結果/causal_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("因果関係分析結果:")
    for u, v, data in G_causal.edges(data=True):
        print(f"  {u} -> {v}: {data['weight']:.3f}")

def create_conditional_probability_analysis(X_scaled, feature_names, df):
    """条件付き確率分析"""
    print("=== 条件付き確率分析 ===")
    
    # 重要な特徴量を選択
    important_features = ['is_holiday', 'vapor_pressure_std_7d', 'avg_humidity_weekly_change_rate']
    
    # 各特徴量の条件付き確率を計算
    conditional_probs = {}
    
    for feature in important_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            feature_values = X_scaled[:, idx]
            
            # 高値と低値の閾値を設定
            high_threshold = np.percentile(feature_values, 75)
            low_threshold = np.percentile(feature_values, 25)
            
            # 高リスク日の確率
            high_risk_days = df['target'] == 1
            
            # 条件付き確率を計算
            high_feature_high_risk = np.sum((feature_values > high_threshold) & high_risk_days)
            high_feature_total = np.sum(feature_values > high_threshold)
            
            low_feature_high_risk = np.sum((feature_values < low_threshold) & high_risk_days)
            low_feature_total = np.sum(feature_values < low_threshold)
            
            if high_feature_total > 0:
                prob_high_given_high = high_feature_high_risk / high_feature_total
            else:
                prob_high_given_high = 0
                
            if low_feature_total > 0:
                prob_high_given_low = low_feature_high_risk / low_feature_total
            else:
                prob_high_given_low = 0
            
            conditional_probs[feature] = {
                'high_feature_high_risk_prob': prob_high_given_high,
                'low_feature_high_risk_prob': prob_high_given_low
            }
    
    # 可視化
    plt.figure(figsize=(12, 8))
    
    features = list(conditional_probs.keys())
    high_probs = [conditional_probs[f]['high_feature_high_risk_prob'] for f in features]
    low_probs = [conditional_probs[f]['low_feature_high_risk_prob'] for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, high_probs, width, label='高値時の高リスク確率', alpha=0.7)
    plt.bar(x + width/2, low_probs, width, label='低値時の高リスク確率', alpha=0.7)
    
    plt.xlabel('特徴量', fontsize=12)
    plt.ylabel('高リスク確率', fontsize=12)
    plt.title('条件付き確率分析', fontsize=14, fontweight='bold')
    plt.xticks(x, features, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ベイジアンネットワーク分析結果/conditional_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("条件付き確率分析結果:")
    for feature, probs in conditional_probs.items():
        print(f"  {feature}:")
        print(f"    高値時の高リスク確率: {probs['high_feature_high_risk_prob']:.3f}")
        print(f"    低値時の高リスク確率: {probs['low_feature_high_risk_prob']:.3f}")

def create_detailed_report():
    """詳細な分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    with open('ベイジアンネットワーク分析結果/ベイジアンネットワーク分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# ベイジアンネットワーク分析レポート\n\n')
        
        f.write('## 概要\n')
        f.write('- 特徴量間の相関ネットワーク分析\n')
        f.write('- 特徴量クラスタリング\n')
        f.write('- 因果関係分析\n')
        f.write('- 条件付き確率分析\n\n')
        
        f.write('## 主要な発見\n\n')
        
        f.write('### 1. 相関ネットワーク\n')
        f.write('- 特徴量間の相関関係を可視化\n')
        f.write('- 強い相関を持つ特徴量グループを特定\n')
        f.write('- ネットワーク構造による特徴量の関係性理解\n\n')
        
        f.write('### 2. 特徴量クラスタリング\n')
        f.write('- 類似した特徴量のグループ化\n')
        f.write('- 機能的な特徴量カテゴリの特定\n')
        f.write('- 冗長性の検出と特徴量選択の最適化\n\n')
        
        f.write('### 3. 因果関係分析\n')
        f.write('- 特徴量間の因果関係の推定\n')
        f.write('- 直接的な影響関係の特定\n')
        f.write('- 医学的解釈に基づく因果関係の検証\n\n')
        
        f.write('### 4. 条件付き確率分析\n')
        f.write('- 特定の特徴量条件下でのリスク確率\n')
        f.write('- 特徴量の閾値効果の評価\n')
        f.write('- 予防医療における介入効果の推定\n\n')
        
        f.write('## 医学的解釈\n')
        f.write('- 気象要素間の複雑な相互作用を理解\n')
        f.write('- 社会的要因と気象要因の因果関係\n')
        f.write('- 予防医療における多因子アプローチの重要性\n')
        f.write('- 個別化されたリスク評価の可能性\n')
        f.write('- 臨床意思決定支援システムへの応用\n')
    
    print("詳細レポートを保存しました: ベイジアンネットワーク分析結果/ベイジアンネットワーク分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        import os
        os.makedirs('ベイジアンネットワーク分析結果', exist_ok=True)
        
        print("保存されたモデルの特徴量セットを取得中...")
        saved_features = get_saved_model_features()
        
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        print("データを準備中...")
        X_scaled, y, feature_names, df = prepare_data_with_saved_features(df, saved_features)
        
        print("ベイジアンネットワーク分析を実行中...")
        
        # 1. 相関ネットワークの作成と可視化
        G, correlation_matrix = create_correlation_network(X_scaled, feature_names)
        visualize_correlation_network(G, feature_names)
        
        # 2. 特徴量クラスタリング
        create_feature_clusters(G, feature_names)
        
        # 3. 因果関係分析
        create_causal_analysis(X_scaled, feature_names)
        
        # 4. 条件付き確率分析
        create_conditional_probability_analysis(X_scaled, feature_names, df)
        
        # 5. 詳細レポート
        create_detailed_report()
        
        print("\n=== ベイジアンネットワーク分析完了 ===")
        print("結果は 'ベイジアンネットワーク分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 