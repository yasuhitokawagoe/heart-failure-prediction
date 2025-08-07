import pandas as pd
import numpy as np
from datetime import datetime

def merge_tokyo_weather_data():
    """東京全体データと気象データを統合"""
    print("=== 東京全体データと気象データの統合開始 ===")

    # 東京全体データの読み込み
    print("東京全体データを読み込み中...")
    tokyo_df = pd.read_csv('東京AMI.csv')
    tokyo_df['date'] = pd.to_datetime(tokyo_df['hospitalization_date'])
    print(f"東京全体データ期間: {tokyo_df['date'].min()} から {tokyo_df['date'].max()}")
    print(f"東京全体データ数: {len(tokyo_df)}")

    # 気象データの読み込み
    print("気象データを読み込み中...")
    weather_df = pd.read_csv('weather_jroad_encoded_2012_2021.csv')
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    print(f"気象データ期間: {weather_df['date'].min()} から {weather_df['date'].max()}")
    print(f"気象データ数: {len(weather_df)}")

    # データの統合
    print("データを統合中...")
    merged_df = pd.merge(tokyo_df, weather_df, on='date', how='inner', suffixes=('_tokyo', '_weather'))

    print(f"統合後のデータ数: {len(merged_df)}")
    print(f"統合後の期間: {merged_df['date'].min()} から {merged_df['date'].max()}")

    # 列名を確認
    print(f"\n統合後の列名（最初の10個）: {merged_df.columns.tolist()[:10]}")

    # 基本統計
    print("\n=== 基本統計 ===")
    print(f"東京全体発生総数: {merged_df['people_tokyo'].sum()}")
    print(f"平均東京全体発生数/日: {merged_df['people_tokyo'].mean():.2f}")
    print(f"最大東京全体発生数/日: {merged_df['people_tokyo'].max()}")
    print(f"最小東京全体発生数/日: {merged_df['people_tokyo'].min()}")

    # 欠損値の確認
    print("\n=== 欠損値確認 ===")
    missing_tokyo = merged_df['people_tokyo'].isnull().sum()
    missing_weather = merged_df[['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].isnull().sum()
    print(f"東京全体データ欠損: {missing_tokyo}")
    print(f"気象データ欠損:")
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        print(f"  {col}: {missing_weather[col]}")

    # データの保存
    print("\n統合データを保存中...")
    merged_df.to_csv('tokyo_weather_merged.csv', index=False)
    print("統合データを保存しました: tokyo_weather_merged.csv")

    # サンプルデータの表示
    print("\n=== サンプルデータ ===")
    print(merged_df[['date', 'people_tokyo', 'avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].head())

    return merged_df

if __name__ == "__main__":
    merged_df = merge_tokyo_weather_data()
    print("\n=== 統合完了 ===") 