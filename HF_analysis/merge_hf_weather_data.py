import pandas as pd
import numpy as np
from datetime import datetime

def merge_hf_weather_data():
    """心不全データと気象データを統合"""
    print("=== 心不全データと気象データの統合開始 ===")

    # 心不全データの読み込み
    print("心不全データを読み込み中...")
    hf_df = pd.read_csv('東京ADHF.csv')
    hf_df['date'] = pd.to_datetime(hf_df['hospitalization_date'])
    print(f"心不全データ期間: {hf_df['date'].min()} から {hf_df['date'].max()}")
    print(f"心不全データ数: {len(hf_df)}")

    # 気象データの読み込み
    print("気象データを読み込み中...")
    weather_df = pd.read_csv('weather_jroad_encoded_2012_2021.csv')
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    print(f"気象データ期間: {weather_df['date'].min()} から {weather_df['date'].max()}")
    print(f"気象データ数: {len(weather_df)}")

    # データの統合
    print("データを統合中...")
    merged_df = pd.merge(hf_df, weather_df, on='date', how='inner', suffixes=('_hf', '_weather'))

    print(f"統合後のデータ数: {len(merged_df)}")
    print(f"統合後の期間: {merged_df['date'].min()} から {merged_df['date'].max()}")

    # 基本統計
    print("\n=== 基本統計 ===")
    print(f"心不全発生総数: {merged_df['people_hf'].sum()}")
    print(f"平均心不全発生数/日: {merged_df['people_hf'].mean():.2f}")
    print(f"最大心不全発生数/日: {merged_df['people_hf'].max()}")
    print(f"最小心不全発生数/日: {merged_df['people_hf'].min()}")

    # 欠損値の確認
    print("\n=== 欠損値確認 ===")
    missing_hf = merged_df['people_hf'].isnull().sum()
    missing_weather = merged_df[['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].isnull().sum()
    print(f"心不全データ欠損: {missing_hf}")
    print(f"気象データ欠損:")
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        print(f"  {col}: {missing_weather[col]}")

    # データの保存
    print("\n統合データを保存中...")
    merged_df.to_csv('hf_weather_merged.csv', index=False)
    print("統合データを保存しました: hf_weather_merged.csv")

    # サンプルデータの表示
    print("\n=== サンプルデータ ===")
    print(merged_df[['date', 'people_hf', 'avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].head())

    return merged_df

if __name__ == "__main__":
    merged_df = merge_hf_weather_data()
    print("\n=== 統合完了 ===") 