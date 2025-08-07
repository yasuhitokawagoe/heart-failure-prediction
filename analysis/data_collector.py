# 疾患別データ収集・統合スクリプト
# 各疾患の結果データを収集して統合データセットを作成

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

class DiseaseDataCollector:
    """疾患別データ収集クラス"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.diseases = {
            'HF': {
                'name': '心不全',
                'path': 'yesterday_work_summary/',
                'data_file': None,  # 実際のファイル名に変更
                'results_file': None
            },
            'AF': {
                'name': '心房細動',
                'path': 'yesterday_work_summary/',
                'data_file': None,
                'results_file': None
            },
            'AMI': {
                'name': '急性心筋梗塞',
                'path': 'yesterday_work_summary/',
                'data_file': 'train_model_weather_only.py',
                'results_file': 'results/detailed_results.json'
            },
            'PE': {
                'name': '肺塞栓症',
                'path': 'yesterday_work_summary/',
                'data_file': None,
                'results_file': None
            },
            'VT_VF': {
                'name': '心室頻拍・心室細動',
                'path': 'yesterday_work_summary/VT_VF_analysis/',
                'data_file': '東京vtvf.csv',
                'results_file': None
            },
            'Tokyo_Total': {
                'name': '東京全体',
                'path': '東京全体_analysis/',
                'data_file': 'tokyo_weather_merged.csv',
                'results_file': 'results/detailed_results.json'
            }
        }
        
        self.collected_data = {}
        
    def collect_all_disease_data(self):
        """全疾患のデータを収集"""
        print("各疾患のデータを収集開始...")
        
        for disease_code, disease_info in self.diseases.items():
            print(f"\n{disease_info['name']}のデータを収集中...")
            
            try:
                # データファイルの読み込み
                data = self._load_disease_data(disease_code, disease_info)
                
                # 結果ファイルの読み込み
                results = self._load_disease_results(disease_code, disease_info)
                
                # 特徴量重要度の読み込み
                feature_importance = self._load_feature_importance(disease_code, disease_info)
                
                self.collected_data[disease_code] = {
                    'name': disease_info['name'],
                    'data': data,
                    'results': results,
                    'feature_importance': feature_importance
                }
                
                print(f"✓ {disease_info['name']}データ収集完了")
                
            except Exception as e:
                print(f"✗ {disease_info['name']}データ収集エラー: {e}")
        
        return self.collected_data
    
    def _load_disease_data(self, disease_code, disease_info):
        """疾患データを読み込み"""
        if disease_info['data_file'] is None:
            return None
            
        data_path = self.base_path / disease_info['path'] / disease_info['data_file']
        
        if data_path.exists():
            if data_path.suffix == '.csv':
                return pd.read_csv(data_path)
            else:
                print(f"未対応のファイル形式: {data_path}")
                return None
        else:
            print(f"データファイルが見つかりません: {data_path}")
            return None
    
    def _load_disease_results(self, disease_code, disease_info):
        """疾患の結果データを読み込み"""
        if disease_info['results_file'] is None:
            return None
            
        results_path = self.base_path / disease_info['path'] / disease_info['results_file']
        
        if results_path.exists():
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"結果ファイル読み込みエラー: {e}")
                return None
        else:
            print(f"結果ファイルが見つかりません: {results_path}")
            return None
    
    def _load_feature_importance(self, disease_code, disease_info):
        """特徴量重要度を読み込み"""
        importance_path = self.base_path / disease_info['path'] / 'results/feature_importance.csv'
        
        if importance_path.exists():
            try:
                importance_df = pd.read_csv(importance_path)
                return importance_df
            except Exception as e:
                print(f"特徴量重要度読み込みエラー: {e}")
                return None
        else:
            print(f"特徴量重要度ファイルが見つかりません: {importance_path}")
            return None
    
    def create_unified_dataset(self):
        """統合データセットを作成"""
        print("\n統合データセットを作成中...")
        
        unified_data = []
        
        for disease_code, disease_data in self.collected_data.items():
            if disease_data['data'] is not None:
                data = disease_data['data'].copy()
                
                # 疾患コードを追加
                data['disease_code'] = disease_code
                data['disease_name'] = disease_data['name']
                
                # 日付列の統一
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                
                # 発症数列の統一（people_tokyo, people_weather, people等）
                people_columns = [col for col in data.columns if 'people' in col.lower()]
                if people_columns:
                    data['incidence_count'] = data[people_columns[0]]
                
                unified_data.append(data)
        
        if unified_data:
            # データフレームを結合
            unified_df = pd.concat(unified_data, ignore_index=True)
            
            # 基本統計情報を追加
            unified_df['year'] = unified_df['date'].dt.year
            unified_df['month'] = unified_df['date'].dt.month
            unified_df['day_of_week'] = unified_df['date'].dt.dayofweek
            
            # 保存
            output_path = Path('data/unified_disease_dataset.csv')
            output_path.parent.mkdir(exist_ok=True)
            unified_df.to_csv(output_path, index=False)
            
            print(f"✓ 統合データセット保存完了: {output_path}")
            print(f"  データ形状: {unified_df.shape}")
            print(f"  疾患数: {unified_df['disease_code'].nunique()}")
            print(f"  期間: {unified_df['date'].min()} 〜 {unified_df['date'].max()}")
            
            return unified_df
        else:
            print("✗ 統合可能なデータが見つかりません")
            return None
    
    def create_summary_statistics(self):
        """サマリー統計を作成"""
        print("\nサマリー統計を作成中...")
        
        summary_stats = {}
        
        for disease_code, disease_data in self.collected_data.items():
            if disease_data['data'] is not None:
                data = disease_data['data']
                
                stats = {
                    'name': disease_data['name'],
                    'data_shape': data.shape,
                    'date_range': {
                        'start': data['date'].min() if 'date' in data.columns else None,
                        'end': data['date'].max() if 'date' in data.columns else None
                    },
                    'weather_features': [col for col in data.columns if 'weather' in col.lower()],
                    'performance_metrics': disease_data['results'] if disease_data['results'] else None
                }
                
                summary_stats[disease_code] = stats
        
        # サマリー統計を保存
        output_path = Path('data/summary_statistics.json')
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ サマリー統計保存完了: {output_path}")
        return summary_stats
    
    def create_weather_comparison_table(self):
        """気象条件比較表を作成"""
        print("\n気象条件比較表を作成中...")
        
        comparison_data = []
        
        for disease_code, disease_data in self.collected_data.items():
            if disease_data['data'] is not None and disease_data['feature_importance'] is not None:
                importance_df = disease_data['feature_importance']
                
                # 気象関連特徴量の重要度を抽出
                weather_importance = importance_df[
                    importance_df['feature'].str.contains('weather|temp|humidity|pressure|wind|sunshine', 
                                                       case=False, na=False)
                ]
                
                for _, row in weather_importance.head(10).iterrows():
                    comparison_data.append({
                        'disease_code': disease_code,
                        'disease_name': disease_data['name'],
                        'feature': row['feature'],
                        'importance': row['importance']
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # 保存
            output_path = Path('data/weather_importance_comparison.csv')
            output_path.parent.mkdir(exist_ok=True)
            comparison_df.to_csv(output_path, index=False)
            
            print(f"✓ 気象条件比較表保存完了: {output_path}")
            return comparison_df
        else:
            print("✗ 比較可能な特徴量重要度データが見つかりません")
            return None
    
    def run_complete_data_collection(self):
        """完全なデータ収集を実行"""
        print("=== 疾患別データ収集プロセス開始 ===")
        
        # 1. 全疾患データの収集
        self.collect_all_disease_data()
        
        # 2. 統合データセットの作成
        unified_dataset = self.create_unified_dataset()
        
        # 3. サマリー統計の作成
        summary_stats = self.create_summary_statistics()
        
        # 4. 気象条件比較表の作成
        weather_comparison = self.create_weather_comparison_table()
        
        print("\n=== データ収集完了 ===")
        print("生成されたファイル:")
        print("- data/unified_disease_dataset.csv")
        print("- data/summary_statistics.json")
        print("- data/weather_importance_comparison.csv")
        
        return {
            'unified_dataset': unified_dataset,
            'summary_stats': summary_stats,
            'weather_comparison': weather_comparison,
            'collected_data': self.collected_data
        }

def main():
    """メイン実行関数"""
    collector = DiseaseDataCollector()
    results = collector.run_complete_data_collection()
    
    print("\n収集結果サマリー:")
    print(f"- 収集対象疾患数: {len(collector.diseases)}")
    print(f"- 成功収集疾患数: {len(collector.collected_data)}")
    
    if results['unified_dataset'] is not None:
        print(f"- 統合データセット形状: {results['unified_dataset'].shape}")

if __name__ == "__main__":
    main() 