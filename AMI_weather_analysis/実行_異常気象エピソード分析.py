# 実行: 異常気象エピソード分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 異常気象エピソード分析モジュールをインポート
from 異常気象エピソード分析 import (
    create_weather_episode_analysis,
    create_episode_visualization,
    create_episode_report,
    analyze_all_samples,
    create_comprehensive_report
)

def load_model_and_data():
    """保存されたモデルとデータを読み込み"""
    try:
        # 保存されたモデルを読み込み
        model_path = 'saved_models/weather_only_results/'
        if os.path.exists(model_path):
            print("保存されたモデルを読み込み中...")
            # モデルの読み込み（実際のファイルパスに応じて調整）
            return None, None
        else:
            print("保存されたモデルが見つかりません")
            return None, None
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return None, None

def load_processed_data():
    """処理済みデータを読み込み"""
    try:
        # データファイルのパスを確認
        data_paths = [
            'yesterday_work_summary/AMI_weather_analysis/saved_models/weather_only_results/',
            'AMI_weather_analysis/saved_models/weather_only_results/',
            'saved_models/weather_only_results/'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                print(f"データパスを発見: {path}")
                # 実際のデータ読み込み処理
                return None
        print("データファイルが見つかりません")
        return None
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def create_demo_episode_analysis():
    """デモンストレーション用のエピソード分析"""
    
    print("デモンストレーション用のエピソード分析を実行します...")
    
    # サンプルデータの作成
    sample_data = {
        'hospitalization_date': ['2023-08-15', '2023-03-20', '2023-12-10', '2023-07-25', '2023-01-15'],
        'hospitalization_count': [3, 0, 2, 4, 1],
        'month': [8, 3, 12, 7, 1],
        'avg_temp': [32.5, 15.2, -2.1, 28.7, 3.5],
        'max_temp': [38.2, 18.5, 2.1, 35.1, 8.2],
        'min_temp': [26.8, 12.1, -5.3, 24.3, -1.2],
        'avg_humidity': [85.2, 65.3, 45.8, 78.9, 72.1],
        'avg_wind': [12.5, 3.2, 8.7, 15.3, 6.8],
        'vapor_pressure': [18.5, 12.3, 8.9, 22.1, 9.8],
        'sunshine_hours': [6.2, 8.5, 4.1, 3.8, 5.9],
        'is_typhoon_condition': [1, 0, 0, 1, 0],
        'is_extremely_hot': [1, 0, 0, 1, 0],
        'is_hot_day': [1, 0, 0, 1, 0],
        'is_tropical_night': [1, 0, 0, 1, 0],
        'is_cold_wave': [0, 0, 1, 0, 1],
        'is_winter_day': [0, 0, 1, 0, 1],
        'is_freezing_day': [0, 0, 1, 0, 0],
        'is_strong_wind': [1, 0, 0, 1, 0],
        'is_rapid_pressure_change': [1, 0, 0, 1, 0],
        'is_extreme_humidity_high': [1, 0, 0, 1, 0],
        'is_extreme_humidity_low': [0, 0, 1, 0, 0],
        'is_rapid_temp_change': [1, 0, 0, 1, 0],
        'is_high_temp_volatility': [1, 0, 0, 1, 0],
        'temp_humidity_stress': [1, 0, 0, 1, 0],
        'pressure_wind_stress': [1, 0, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    df['hospitalization_date'] = pd.to_datetime(df['hospitalization_date'])
    
    # 予測確率のサンプル
    predictions = [0.892, 0.156, 0.734, 0.945, 0.623]
    
    # エピソード分析の実行
    output_dir = '異常気象エピソード分析結果_デモ'
    os.makedirs(output_dir, exist_ok=True)
    
    all_episodes = []
    
    for i in range(len(df)):
        print(f"エピソード分析中: サンプル {i+1}/5")
        
        # エピソード分析
        episode_summary = create_weather_episode_analysis(i, df, predictions)
        all_episodes.append(episode_summary)
        
        # 可視化
        viz_path = os.path.join(output_dir, f'episode_analysis_sample_{i}.png')
        create_episode_visualization(episode_summary, viz_path)
        
        # レポート
        report_path = os.path.join(output_dir, f'episode_report_sample_{i}.md')
        create_episode_report(episode_summary, report_path)
        
        # コンソール出力
        print(f"=== サンプル {i+1} エピソード分析 ===")
        print(f"日付: {episode_summary['date']}")
        print(f"予測確率: {episode_summary['prediction']:.3f}")
        print(f"リスクレベル: {episode_summary['risk_level']}")
        print(f"エピソード重症度: {episode_summary['episode_severity']}")
        print(f"アクティブエピソード: {len(episode_summary['active_episodes'])}個")
        for episode in episode_summary['active_episodes']:
            print(f"  - {episode}")
        print()
    
    # 総合レポート
    create_comprehensive_report(all_episodes, output_dir)
    
    print(f"デモンストレーション完了")
    print(f"結果は '{output_dir}/' ディレクトリに保存されました")
    
    return all_episodes

def create_sample_episode_summary():
    """サンプルエピソードサマリーの作成"""
    
    print("サンプルエピソードサマリーを作成します...")
    
    # 高リスクケースのサンプル
    high_risk_episode = {
        'date': '2023年8月15日',
        'prediction': 0.892,
        'actual_count': 3,
        'season': '夏',
        'weather_data': {
            'avg_temp': 32.5,
            'max_temp': 38.2,
            'min_temp': 26.8,
            'avg_humidity': 85.2,
            'avg_wind': 12.5,
            'vapor_pressure': 18.5,
            'sunshine_hours': 6.2
        },
        'active_episodes': [
            '台風接近による気圧急変と強風',
            '猛暑日による熱ストレス',
            '真夏日による暑さ',
            '熱帯夜による睡眠不足',
            '強風による気象ストレス',
            '急激な気圧変化',
            '極端な高湿度',
            '急激な気温変化',
            '高気温変動性',
            '高温多湿の複合ストレス',
            '低気圧強風の複合ストレス'
        ],
        'episode_severity': 15,
        'risk_level': '極高リスク'
    }
    
    # 低リスクケースのサンプル
    low_risk_episode = {
        'date': '2023年3月20日',
        'prediction': 0.156,
        'actual_count': 0,
        'season': '春',
        'weather_data': {
            'avg_temp': 15.2,
            'max_temp': 18.5,
            'min_temp': 12.1,
            'avg_humidity': 65.3,
            'avg_wind': 3.2,
            'vapor_pressure': 12.3,
            'sunshine_hours': 8.5
        },
        'active_episodes': [],
        'episode_severity': 0,
        'risk_level': '低リスク'
    }
    
    # サマリーレポートの作成
    output_dir = '異常気象エピソード分析結果_サンプル'
    os.makedirs(output_dir, exist_ok=True)
    
    # 高リスクケースの可視化とレポート
    create_episode_visualization(high_risk_episode, 
                                os.path.join(output_dir, 'high_risk_episode.png'))
    create_episode_report(high_risk_episode, 
                         os.path.join(output_dir, 'high_risk_episode.md'))
    
    # 低リスクケースの可視化とレポート
    create_episode_visualization(low_risk_episode, 
                                os.path.join(output_dir, 'low_risk_episode.png'))
    create_episode_report(low_risk_episode, 
                         os.path.join(output_dir, 'low_risk_episode.md'))
    
    # 比較レポート
    with open(os.path.join(output_dir, 'エピソード比較レポート.md'), 'w', encoding='utf-8') as f:
        f.write('# 異常気象エピソード比較レポート\n\n')
        
        f.write('## 高リスクケース vs 低リスクケース\n\n')
        
        f.write('### 高リスクケース (2023年8月15日)\n')
        f.write(f'- **予測確率**: {high_risk_episode["prediction"]:.3f}\n')
        f.write(f'- **リスクレベル**: {high_risk_episode["risk_level"]}\n')
        f.write(f'- **エピソード重症度**: {high_risk_episode["episode_severity"]}\n')
        f.write(f'- **アクティブエピソード数**: {len(high_risk_episode["active_episodes"])}\n')
        f.write(f'- **季節**: {high_risk_episode["season"]}\n\n')
        
        f.write('**主要な異常気象エピソード**:\n')
        for episode in high_risk_episode['active_episodes'][:5]:  # 上位5個
            f.write(f'- {episode}\n')
        f.write('\n')
        
        f.write('### 低リスクケース (2023年3月20日)\n')
        f.write(f'- **予測確率**: {low_risk_episode["prediction"]:.3f}\n')
        f.write(f'- **リスクレベル**: {low_risk_episode["risk_level"]}\n')
        f.write(f'- **エピソード重症度**: {low_risk_episode["episode_severity"]}\n')
        f.write(f'- **アクティブエピソード数**: {len(low_risk_episode["active_episodes"])}\n')
        f.write(f'- **季節**: {low_risk_episode["season"]}\n\n')
        
        f.write('**気象状況**: 異常気象エピソードなし - 安定した春の気候\n\n')
        
        f.write('## 主要な発見\n')
        f.write('1. **異常気象の複合効果**: 複数の異常気象が同時発生することでリスクが大幅に増加\n')
        f.write('2. **季節性の影響**: 夏の異常気象（台風+猛暑）が特にリスクを増加\n')
        f.write('3. **気象ストレスの累積**: 11個の異常気象エピソードが同時発生\n')
        f.write('4. **予測精度**: エピソード重症度と予測確率の高い相関\n\n')
        
        f.write('## 臨床的意義\n')
        f.write('- **予防医療**: 異常気象予報に基づく事前警告システム\n')
        f.write('- **個別化ケア**: 患者ごとの気象感受性評価\n')
        f.write('- **医療体制**: 高リスク日の医療資源強化\n')
        f.write('- **患者教育**: 具体的な気象状況での注意喚起\n')
    
    print(f"サンプルエピソードサマリー完了")
    print(f"結果は '{output_dir}/' ディレクトリに保存されました")

def main():
    """メイン実行関数"""
    try:
        print("=== 異常気象エピソード分析実行 ===")
        
        # 1. デモンストレーション実行
        print("\n1. デモンストレーション実行")
        demo_episodes = create_demo_episode_analysis()
        
        # 2. サンプルエピソードサマリー作成
        print("\n2. サンプルエピソードサマリー作成")
        create_sample_episode_summary()
        
        print("\n=== 異常気象エピソード分析完了 ===")
        print("結果は以下のディレクトリに保存されました:")
        print("- 異常気象エピソード分析結果_デモ/")
        print("- 異常気象エピソード分析結果_サンプル/")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 