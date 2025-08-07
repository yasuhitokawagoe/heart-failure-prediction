import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class IntegratedNetworkAnalysis:
    def __init__(self):
        self.weather_data = None
        self.disease_data = None
        self.lag_features = None
        self.network_data = None
        
    def load_and_prepare_data(self):
        """データの読み込みと前処理"""
        print("=== データ読み込みと前処理 ===")
        
        # 気象データの読み込み
        self.weather_data = pd.read_csv('data/tokyo_weather_merged.csv')
        self.weather_data['date'] = pd.to_datetime(self.weather_data['date'])
        self.weather_data.set_index('date', inplace=True)
        
        # 疾患データの読み込み
        diseases = {
            'AF': 'data/東京AF.csv',
            'HF': 'data/東京ADHF.csv', 
            'AMI': 'data/東京AMI.csv',
            'VT_VF': 'data/東京vtvf.csv',
            'PE': 'data/東京PE.csv'
        }
        
        self.disease_data = {}
        for disease, file_path in diseases.items():
            data = pd.read_csv(file_path)
            if disease == 'AF':
                data['date'] = pd.to_datetime(data['hospitalization_date_af'])
                data = data.groupby('date')['people_af'].sum().reset_index()
                data.columns = ['date', f'{disease}_incidence']
            else:
                data['date'] = pd.to_datetime(data['hospitalization_date'])
                data = data.groupby('date')['people'].sum().reset_index()
                data.columns = ['date', f'{disease}_incidence']
            self.disease_data[disease] = data
        
        print("✓ データ読み込み完了")
        
    def create_lag_features(self):
        """ラグ特徴量の作成"""
        print("\n=== ラグ特徴量作成 ===")
        
        # 基本気象データのラグ特徴量
        weather_cols = ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local', 
                       'avg_wind_weather', 'sunshine_hours_weather']
        
        self.lag_features = self.weather_data.copy()
        
        # 1-7日間のラグ特徴量
        for col in weather_cols:
            for lag in [1, 2, 3, 7]:
                self.lag_features[f'{col}_lag_{lag}'] = self.lag_features[col].shift(lag)
        
        # 変化率の計算
        for col in weather_cols:
            self.lag_features[f'{col}_change_1d'] = (
                self.lag_features[col] - self.lag_features[f'{col}_lag_1']
            ) / self.lag_features[f'{col}_lag_1']
            
            self.lag_features[f'{col}_change_3d'] = (
                self.lag_features[col] - self.lag_features[f'{col}_lag_3']
            ) / self.lag_features[f'{col}_lag_3']
            
            self.lag_features[f'{col}_change_7d'] = (
                self.lag_features[col] - self.lag_features[f'{col}_lag_7']
            ) / self.lag_features[f'{col}_lag_7']
        
        # 移動平均とボラティリティ
        for col in weather_cols:
            self.lag_features[f'{col}_ma_3d'] = self.lag_features[col].rolling(window=3).mean()
            self.lag_features[f'{col}_ma_7d'] = self.lag_features[col].rolling(window=7).mean()
            self.lag_features[f'{col}_volatility_3d'] = self.lag_features[col].rolling(window=3).std()
            self.lag_features[f'{col}_volatility_7d'] = self.lag_features[col].rolling(window=7).std()
        
        # 異常気象フラグの作成
        self.create_extreme_weather_flags()
        
        print("✓ ラグ特徴量作成完了")
        
    def create_extreme_weather_flags(self):
        """異常気象フラグの作成"""
        print("異常気象フラグを作成中...")
        
        # 温度関連フラグ
        temp_col = 'avg_temp_weather'
        temp_quantiles = self.lag_features[temp_col].quantile([0.1, 0.9])
        
        self.lag_features['is_extreme_cold'] = (self.lag_features[temp_col] < temp_quantiles[0.1]).astype(int)
        self.lag_features['is_extreme_hot'] = (self.lag_features[temp_col] > temp_quantiles[0.9]).astype(int)
        self.lag_features['is_temp_volatile'] = (self.lag_features[f'{temp_col}_volatility_3d'] > 
                                               self.lag_features[f'{temp_col}_volatility_3d'].quantile(0.8)).astype(int)
        
        # 湿度関連フラグ
        humidity_col = 'avg_humidity_weather'
        humidity_quantiles = self.lag_features[humidity_col].quantile([0.1, 0.9])
        
        self.lag_features['is_extreme_dry'] = (self.lag_features[humidity_col] < humidity_quantiles[0.1]).astype(int)
        self.lag_features['is_extreme_humid'] = (self.lag_features[humidity_col] > humidity_quantiles[0.9]).astype(int)
        
        # 気圧関連フラグ
        pressure_col = 'pressure_local'
        pressure_quantiles = self.lag_features[pressure_col].quantile([0.1, 0.9])
        
        self.lag_features['is_low_pressure'] = (self.lag_features[pressure_col] < pressure_quantiles[0.1]).astype(int)
        self.lag_features['is_high_pressure'] = (self.lag_features[pressure_col] > pressure_quantiles[0.9]).astype(int)
        self.lag_features['is_pressure_volatile'] = (self.lag_features[f'{pressure_col}_volatility_3d'] > 
                                                   self.lag_features[f'{pressure_col}_volatility_3d'].quantile(0.8)).astype(int)
        
        # 風速関連フラグ
        wind_col = 'avg_wind_weather'
        wind_quantiles = self.lag_features[wind_col].quantile([0.8, 0.9])
        
        self.lag_features['is_high_wind'] = (self.lag_features[wind_col] > wind_quantiles[0.8]).astype(int)
        self.lag_features['is_extreme_wind'] = (self.lag_features[wind_col] > wind_quantiles[0.9]).astype(int)
        
        # 日照時間関連フラグ
        sunshine_col = 'sunshine_hours_weather'
        sunshine_quantiles = self.lag_features[sunshine_col].quantile([0.1, 0.9])
        
        self.lag_features['is_low_sunshine'] = (self.lag_features[sunshine_col] < sunshine_quantiles[0.1]).astype(int)
        self.lag_features['is_high_sunshine'] = (self.lag_features[sunshine_col] > sunshine_quantiles[0.9]).astype(int)
        
        # 複合フラグ
        self.lag_features['is_cold_wave'] = (self.lag_features['is_extreme_cold'] & 
                                           (self.lag_features[f'{temp_col}_lag_1'] < temp_quantiles[0.1])).astype(int)
        
        self.lag_features['is_heat_wave'] = (self.lag_features['is_extreme_hot'] & 
                                           (self.lag_features[f'{temp_col}_lag_1'] > temp_quantiles[0.9])).astype(int)
        
        self.lag_features['is_humid_hot'] = (self.lag_features['is_extreme_humid'] & 
                                           self.lag_features['is_extreme_hot']).astype(int)
        
        self.lag_features['is_dry_cold'] = (self.lag_features['is_extreme_dry'] & 
                                          self.lag_features['is_extreme_cold']).astype(int)
        
        print("✓ 異常気象フラグ作成完了")
        
    def align_disease_data(self):
        """疾患データの整列"""
        print("\n=== 疾患データ整列 ===")
        
        # 日付範囲を統一
        all_dates = set(self.lag_features.index)
        for disease_data in self.disease_data.values():
            all_dates.update(disease_data['date'])
        
        # 完全な日付範囲を作成
        date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
        master_df = pd.DataFrame({'date': date_range})
        master_df.set_index('date', inplace=True)
        
        # 気象データをマージ
        master_df = master_df.join(self.lag_features, how='left')
        
        # 疾患データをマージ
        for disease, data in self.disease_data.items():
            data_temp = data.copy()
            data_temp.set_index('date', inplace=True)
            master_df = master_df.join(data_temp, how='left')
            master_df[f'{disease}_incidence'] = master_df[f'{disease}_incidence'].fillna(0)
        
        self.network_data = master_df
        print(f"✓ データ整列完了: {len(self.network_data)}日間のデータ")
        
    def analyze_weather_disease_correlations(self):
        """気象-疾患相関分析"""
        print("\n=== 気象-疾患相関分析 ===")
        
        # 気象特徴量の選択
        weather_features = [col for col in self.network_data.columns if any(x in col for x in 
                          ['avg_temp', 'avg_humidity', 'pressure', 'avg_wind', 'sunshine', 
                           'lag_', 'change_', 'volatility_', 'is_'])]
        
        # 疾患特徴量の選択
        disease_features = [col for col in self.network_data.columns if 'incidence' in col]
        
        # 相関行列の計算
        correlation_data = self.network_data[weather_features + disease_features].corr()
        
        # 気象-疾患相関の抽出
        weather_disease_corr = correlation_data.loc[weather_features, disease_features]
        
        # 上位相関の抽出
        top_correlations = []
        for disease in disease_features:
            for weather in weather_features:
                corr_value = weather_disease_corr.loc[weather, disease]
                if abs(corr_value) > 0.1:  # 相関係数0.1以上
                    top_correlations.append({
                        'weather_feature': weather,
                        'disease': disease,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        top_correlations_df = pd.DataFrame(top_correlations)
        top_correlations_df = top_correlations_df.sort_values('abs_correlation', ascending=False)
        
        print(f"✓ 相関分析完了: {len(top_correlations_df)}個の有意な相関を発見")
        
        return weather_disease_corr, top_correlations_df
        
    def create_network_visualization(self, top_correlations_df):
        """ネットワーク可視化の作成"""
        print("\n=== ネットワーク可視化作成 ===")
        
        # NetworkXグラフの作成
        G = nx.Graph()
        
        # ノードの追加
        weather_nodes = set(top_correlations_df['weather_feature'].unique())
        disease_nodes = set(top_correlations_df['disease'].unique())
        
        for node in weather_nodes:
            G.add_node(node, type='weather', layer=1)
        
        for node in disease_nodes:
            G.add_node(node, type='disease', layer=2)
        
        # エッジの追加
        for _, row in top_correlations_df.iterrows():
            weight = abs(row['correlation'])
            if weight > 0.1:  # 相関係数0.1以上のみ
                G.add_edge(row['weather_feature'], row['disease'], 
                          weight=weight, correlation=row['correlation'])
        
        # レイアウトの計算
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 可視化
        plt.figure(figsize=(15, 10))
        
        # ノードの描画
        weather_pos = {node: pos[node] for node in weather_nodes}
        disease_pos = {node: pos[node] for node in disease_nodes}
        
        # 気象ノード（青色）
        nx.draw_networkx_nodes(G, pos, nodelist=weather_nodes, 
                              node_color='lightblue', node_size=1000, alpha=0.7)
        
        # 疾患ノード（赤色）
        nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, 
                              node_color='lightcoral', node_size=1500, alpha=0.8)
        
        # エッジの描画
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]  # エッジの太さ
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
        
        # ラベルの描画
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Weather-Disease Network Analysis\n(Edge thickness = correlation strength)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('visualizations/weather_disease_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ ネットワーク可視化保存: visualizations/weather_disease_network.png")
        
        return G
        
    def create_interactive_dashboard(self, top_correlations_df):
        """インタラクティブダッシュボードの作成"""
        print("\n=== インタラクティブダッシュボード作成 ===")
        
        # 相関ヒートマップ
        weather_features = top_correlations_df['weather_feature'].unique()[:20]  # 上位20個
        disease_features = top_correlations_df['disease'].unique()
        
        # 相関行列の作成
        corr_matrix = self.network_data[list(weather_features) + list(disease_features)].corr()
        weather_disease_corr = corr_matrix.loc[weather_features, disease_features]
        
        # Plotlyヒートマップ
        fig = go.Figure(data=go.Heatmap(
            z=weather_disease_corr.values,
            x=weather_disease_corr.columns,
            y=weather_disease_corr.index,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(weather_disease_corr.values, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Weather-Disease Correlation Heatmap',
            xaxis_title='Diseases',
            yaxis_title='Weather Features',
            width=1200,
            height=800
        )
        
        fig.write_html('visualizations/weather_disease_heatmap.html')
        print("✓ インタラクティブヒートマップ保存: visualizations/weather_disease_heatmap.html")
        
        return fig
        
    def create_lag_analysis(self):
        """ラグ特徴量の分析"""
        print("\n=== ラグ特徴量分析 ===")
        
        # ラグ特徴量の効果分析
        lag_features = [col for col in self.network_data.columns if 'lag_' in col]
        disease_features = [col for col in self.network_data.columns if 'incidence' in col]
        
        lag_effects = {}
        for disease in disease_features:
            lag_effects[disease] = {}
            for lag_feature in lag_features:
                # 両方のデータからNaNを除去し、同じ長さにする
                disease_data = self.network_data[disease].dropna()
                lag_data = self.network_data[lag_feature].dropna()
                
                # 共通のインデックスでデータを整列
                common_index = disease_data.index.intersection(lag_data.index)
                if len(common_index) > 10:  # 最低10個のデータポイントが必要
                    disease_aligned = disease_data.loc[common_index]
                    lag_aligned = lag_data.loc[common_index]
                    
                    try:
                        corr, p_val = pearsonr(disease_aligned, lag_aligned)
                        lag_effects[disease][lag_feature] = {'correlation': corr, 'p_value': p_val}
                    except:
                        lag_effects[disease][lag_feature] = {'correlation': 0, 'p_value': 1}
                else:
                    lag_effects[disease][lag_feature] = {'correlation': 0, 'p_value': 1}
        
        # 上位ラグ効果の抽出
        top_lag_effects = []
        for disease, effects in lag_effects.items():
            for lag_feature, stats in effects.items():
                if abs(stats['correlation']) > 0.1 and stats['p_value'] < 0.05:
                    top_lag_effects.append({
                        'disease': disease,
                        'lag_feature': lag_feature,
                        'correlation': stats['correlation'],
                        'p_value': stats['p_value'],
                        'abs_correlation': abs(stats['correlation'])
                    })
        
        top_lag_effects_df = pd.DataFrame(top_lag_effects)
        if len(top_lag_effects_df) > 0:
            top_lag_effects_df = top_lag_effects_df.sort_values('abs_correlation', ascending=False)
        
        print(f"✓ ラグ分析完了: {len(top_lag_effects_df)}個の有意なラグ効果を発見")
        
        return lag_effects, top_lag_effects_df
        
    def generate_report(self, top_correlations_df, top_lag_effects_df):
        """分析レポートの生成"""
        print("\n=== 分析レポート生成 ===")
        
        # データサマリーの計算
        weather_features_count = len([col for col in self.network_data.columns if any(x in col for x in 
                                    ['avg_temp', 'avg_humidity', 'pressure', 'avg_wind', 'sunshine'])])
        lag_features_count = len([col for col in self.network_data.columns if 'lag_' in col])
        extreme_weather_flags_count = len([col for col in self.network_data.columns if 'is_' in col])
        diseases_count = len([col for col in self.network_data.columns if 'incidence' in col])
        
        # 上位相関の情報
        strongest_pair = None
        most_affected_disease = None
        most_influential_weather = None
        
        if len(top_correlations_df) > 0:
            strongest_pair = top_correlations_df.iloc[0].to_dict()
            
            # 疾患別平均相関
            disease_avg_corr = top_correlations_df.groupby('disease')['abs_correlation'].mean()
            if len(disease_avg_corr) > 0:
                most_affected_disease = disease_avg_corr.idxmax()
            
            # 気象特徴量別平均相関
            weather_avg_corr = top_correlations_df.groupby('weather_feature')['abs_correlation'].mean()
            if len(weather_avg_corr) > 0:
                most_influential_weather = weather_avg_corr.idxmax()
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_days': len(self.network_data),
                'weather_features': weather_features_count,
                'lag_features': lag_features_count,
                'extreme_weather_flags': extreme_weather_flags_count,
                'diseases': diseases_count
            },
            'top_correlations': top_correlations_df.head(20).to_dict('records') if len(top_correlations_df) > 0 else [],
            'top_lag_effects': top_lag_effects_df.head(20).to_dict('records') if len(top_lag_effects_df) > 0 else [],
            'weather_disease_summary': {
                'strongest_weather_disease_pair': strongest_pair,
                'most_affected_disease': most_affected_disease,
                'most_influential_weather': most_influential_weather
            }
        }
        
        # レポートの保存
        with open('reports/integrated_network_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print("✓ 分析レポート保存: reports/integrated_network_analysis_report.json")
        
        return report
        
    def run_analysis(self):
        """完全な分析の実行"""
        print("=== 統合ネットワーク分析開始 ===")
        
        # 1. データ読み込み
        self.load_and_prepare_data()
        
        # 2. ラグ特徴量作成
        self.create_lag_features()
        
        # 3. 疾患データ整列
        self.align_disease_data()
        
        # 4. 相関分析
        weather_disease_corr, top_correlations_df = self.analyze_weather_disease_correlations()
        
        # 5. ネットワーク可視化
        G = self.create_network_visualization(top_correlations_df)
        
        # 6. インタラクティブダッシュボード
        fig = self.create_interactive_dashboard(top_correlations_df)
        
        # 7. ラグ分析
        lag_effects, top_lag_effects_df = self.create_lag_analysis()
        
        # 8. レポート生成
        report = self.generate_report(top_correlations_df, top_lag_effects_df)
        
        print("\n=== 統合ネットワーク分析完了 ===")
        print(f"✓ 発見された有意な相関: {len(top_correlations_df)}個")
        print(f"✓ 発見された有意なラグ効果: {len(top_lag_effects_df)}個")
        
        return {
            'network_graph': G,
            'top_correlations': top_correlations_df,
            'top_lag_effects': top_lag_effects_df,
            'report': report
        }

if __name__ == "__main__":
    # 分析の実行
    analyzer = IntegratedNetworkAnalysis()
    results = analyzer.run_analysis()
    
    print("\n=== 分析結果サマリー ===")
    if len(results['top_correlations']) > 0:
        print(f"最も強い気象-疾患相関: {results['top_correlations'].iloc[0]}")
    else:
        print("最も強い気象-疾患相関: なし")
        
    if len(results['top_lag_effects']) > 0:
        print(f"最も強いラグ効果: {results['top_lag_effects'].iloc[0]}")
    else:
        print("最も強いラグ効果: なし") 