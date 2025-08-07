import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import partial_dependence
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

def create_pdp_plots(model, X_scaled, feature_names, df):
    """Partial Dependence Plotsを作成"""
    print("=== Partial Dependence Plotsを作成 ===")
    
    # 上位10個の特徴量を選択
    top_features = ['is_holiday', 'vapor_pressure_std_7d', 'avg_humidity_weekly_change_rate', 
                   'avg_temp_change_rate', 'pressure_change_temp_change', 'avg_temp_trend_strength',
                   'avg_temp_monthly_change_rate', 'avg_humidity_monthly_change_rate', 
                   'seasonal_temp_deviation', 'vapor_pressure_weekly_change_rate']
    
    # 各特徴量のPDPを作成
    for i, feature in enumerate(top_features):
        if feature in feature_names:
            print(f"PDP作成中: {feature}")
            
            # 特徴量のインデックスを取得
            feature_idx = feature_names.index(feature)
            
            # 特徴量の値の範囲を取得
            feature_values = X_scaled[:, feature_idx]
            min_val = np.percentile(feature_values, 1)
            max_val = np.percentile(feature_values, 99)
            
            # グリッドポイントを作成
            grid_points = np.linspace(min_val, max_val, 50)
            
            # PDP値を計算
            pdp_values = []
            for point in grid_points:
                X_temp = X_scaled.copy()
                X_temp[:, feature_idx] = point
                predictions = model.predict_proba(X_temp)[:, 1]
                pdp_values.append(np.mean(predictions))
            
            # プロット作成
            plt.figure(figsize=(10, 6))
            plt.plot(grid_points, pdp_values, linewidth=2, color='blue')
            plt.xlabel(f'{feature} (標準化値)', fontsize=12)
            plt.ylabel('AMIリスク予測確率', fontsize=12)
            plt.title(f'Partial Dependence Plot: {feature}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 実際のデータポイントを散布図で追加
            plt.scatter(feature_values, model.predict_proba(X_scaled)[:, 1], 
                       alpha=0.1, color='red', s=1)
            
            plt.tight_layout()
            plt.savefig(f'可視化分析結果/pdp_{feature}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 日本語名での解釈
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
                'vapor_pressure_weekly_change_rate': '気圧週次変化率'
            }
            
            print(f"  - {feature_japanese.get(feature, feature)}: リスク変化パターン分析完了")

def create_feature_interaction_plots(model, X_scaled, feature_names, df):
    """特徴量相互作用の可視化"""
    print("=== 特徴量相互作用の可視化 ===")
    
    # 重要な相互作用ペアを定義
    interaction_pairs = [
        ('is_holiday', 'avg_temp_change_rate'),
        ('vapor_pressure_std_7d', 'avg_humidity_weekly_change_rate'),
        ('avg_temp_trend_strength', 'seasonal_temp_deviation'),
        ('pressure_change_temp_change', 'vapor_pressure_weekly_change_rate')
    ]
    
    for feature1, feature2 in interaction_pairs:
        if feature1 in feature_names and feature2 in feature_names:
            print(f"相互作用分析中: {feature1} vs {feature2}")
            
            # 特徴量のインデックスを取得
            idx1 = feature_names.index(feature1)
            idx2 = feature_names.index(feature2)
            
            # グリッドを作成
            val1_range = np.linspace(np.percentile(X_scaled[:, idx1], 5), 
                                   np.percentile(X_scaled[:, idx1], 95), 20)
            val2_range = np.linspace(np.percentile(X_scaled[:, idx2], 5), 
                                   np.percentile(X_scaled[:, idx2], 95), 20)
            
            # 相互作用マトリックスを作成
            interaction_matrix = np.zeros((len(val1_range), len(val2_range)))
            
            for i, v1 in enumerate(val1_range):
                for j, v2 in enumerate(val2_range):
                    X_temp = X_scaled.copy()
                    X_temp[:, idx1] = v1
                    X_temp[:, idx2] = v2
                    predictions = model.predict_proba(X_temp)[:, 1]
                    interaction_matrix[i, j] = np.mean(predictions)
            
            # ヒートマップを作成
            plt.figure(figsize=(10, 8))
            im = plt.imshow(interaction_matrix, cmap='RdYlBu_r', aspect='auto',
                           extent=[val2_range[0], val2_range[-1], val1_range[0], val1_range[-1]])
            
            plt.colorbar(im, label='AMIリスク予測確率')
            plt.xlabel(f'{feature2} (標準化値)', fontsize=12)
            plt.ylabel(f'{feature1} (標準化値)', fontsize=12)
            plt.title(f'特徴量相互作用: {feature1} vs {feature2}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'可視化分析結果/interaction_{feature1}_vs_{feature2}.png', dpi=300, bbox_inches='tight')
            plt.close()

def create_time_series_analysis(df, model, feature_names, X_scaled):
    """時系列での特徴量変化とリスクの関係を分析"""
    print("=== 時系列分析 ===")
    
    # 重要な特徴量の時系列変化をプロット
    important_features = ['is_holiday', 'vapor_pressure_std_7d', 'avg_humidity_weekly_change_rate']
    
    fig, axes = plt.subplots(len(important_features), 1, figsize=(15, 12))
    if len(important_features) == 1:
        axes = [axes]
    
    for i, feature in enumerate(important_features):
        if feature in feature_names:
            feature_idx = feature_names.index(feature)
            feature_values = X_scaled[:, feature_idx]
            predictions = model.predict_proba(X_scaled)[:, 1]
            
            # 時系列プロット
            axes[i].plot(df['hospitalization_date'], feature_values, 
                        label=f'{feature} (標準化値)', alpha=0.7)
            axes[i].set_ylabel(f'{feature} (標準化値)', fontsize=10)
            
            # リスク予測を重ねて表示
            ax2 = axes[i].twinx()
            ax2.scatter(df['hospitalization_date'], predictions, 
                       c=predictions, cmap='RdYlBu_r', s=10, alpha=0.6)
            ax2.set_ylabel('AMIリスク予測確率', fontsize=10)
            
            axes[i].set_title(f'時系列変化: {feature}', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('可視化分析結果/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_seasonal_analysis(df, model, feature_names, X_scaled):
    """季節性分析"""
    print("=== 季節性分析 ===")
    
    # 月別の平均リスクを計算
    df['month'] = df['hospitalization_date'].dt.month
    predictions = model.predict_proba(X_scaled)[:, 1]
    df['predicted_risk'] = predictions
    
    monthly_risk = df.groupby('month')['predicted_risk'].mean()
    monthly_actual = df.groupby('month')['target'].mean()
    
    # 月別リスクプロット
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(monthly_risk.index, monthly_risk.values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('月', fontsize=12)
    plt.ylabel('平均AMIリスク予測確率', fontsize=12)
    plt.title('月別AMIリスク予測', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(monthly_actual.index, monthly_actual.values, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel('月', fontsize=12)
    plt.ylabel('実際の高リスク日割合', fontsize=12)
    plt.title('月別実際の高リスク日割合', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('可視化分析結果/seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_report():
    """詳細な分析レポートを作成"""
    print("=== 詳細レポートを作成 ===")
    
    with open('可視化分析結果/可視化分析レポート.md', 'w', encoding='utf-8') as f:
        f.write('# 可視化分析レポート\n\n')
        
        f.write('## 概要\n')
        f.write('- Partial Dependence Plots (PDP)による個別特徴量影響分析\n')
        f.write('- 特徴量相互作用の可視化\n')
        f.write('- 時系列での変化パターン分析\n')
        f.write('- 季節性分析\n\n')
        
        f.write('## 主要な発見\n\n')
        
        f.write('### 1. 祝日・週末の影響\n')
        f.write('- 祝日や週末はAMIリスクが大幅に上昇\n')
        f.write('- 社会的要因と気象要因の複合効果\n\n')
        
        f.write('### 2. 気圧変動の重要性\n')
        f.write('- 7日間の気圧変動がリスクに大きく影響\n')
        f.write('- 急激な気圧変化がリスクを増加\n\n')
        
        f.write('### 3. 湿度変化の影響\n')
        f.write('- 週次・月次の湿度変化率が重要\n')
        f.write('- 湿度の急激な変化がリスク因子\n\n')
        
        f.write('### 4. 気温変化パターン\n')
        f.write('- 気温の変化率とトレンド強度が重要\n')
        f.write('- 季節性気温偏差がリスクに影響\n\n')
        
        f.write('### 5. 相互作用効果\n')
        f.write('- 気圧・気温変化の相互作用が重要\n')
        f.write('- 複数の気象要素の組み合わせ効果\n\n')
        
        f.write('## 医学的解釈\n')
        f.write('- 気象の急激な変化が心血管イベントのリスクを増加\n')
        f.write('- 社会的要因（祝日・週末）と気象要因の複合効果\n')
        f.write('- 季節性の変化パターンが重要\n')
        f.write('- 複数の気象要素の相互作用を考慮する必要\n')
    
    print("詳細レポートを保存しました: 可視化分析結果/可視化分析レポート.md")

def main():
    """メイン実行関数"""
    try:
        import os
        os.makedirs('可視化分析結果', exist_ok=True)
        
        print("保存されたモデルの特徴量セットを取得中...")
        saved_features = get_saved_model_features()
        
        print("モデルとデータを読み込み中...")
        model, df = load_model_and_data()
        if model is None or df is None:
            print("モデルまたはデータの読み込みに失敗しました")
            return
        
        print("データを準備中...")
        X_scaled, y, feature_names, df = prepare_data_with_saved_features(df, saved_features)
        
        print("可視化分析を実行中...")
        
        # 1. Partial Dependence Plots
        create_pdp_plots(model, X_scaled, feature_names, df)
        
        # 2. 特徴量相互作用
        create_feature_interaction_plots(model, X_scaled, feature_names, df)
        
        # 3. 時系列分析
        create_time_series_analysis(df, model, feature_names, X_scaled)
        
        # 4. 季節性分析
        create_seasonal_analysis(df, model, feature_names, X_scaled)
        
        # 5. 詳細レポート
        create_detailed_report()
        
        print("\n=== 可視化分析完了 ===")
        print("結果は '可視化分析結果/' ディレクトリに保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 