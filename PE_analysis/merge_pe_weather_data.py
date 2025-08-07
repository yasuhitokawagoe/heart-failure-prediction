import pandas as pd
import numpy as np
from datetime import datetime

def merge_pe_weather_data():
    """PEデータと気象データを統合"""
    print("=== PEデータと気象データの統合開始 ===")
    
    # PEデータの読み込み
    print("PEデータを読み込み中...")
    pe_df = pd.read_csv('東京PE.csv')
    pe_df['date'] = pd.to_datetime(pe_df['hospitalization_date'])
    print(f"PEデータ期間: {pe_df['date'].min()} から {pe_df['date'].max()}")
    print(f"PEデータ数: {len(pe_df)}")
    
    # 気象データの読み込み
    print("気象データを読み込み中...")
    weather_df = pd.read_csv('weather_jroad_encoded_2012_2021.csv')
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    print(f"気象データ期間: {weather_df['date'].min()} から {weather_df['date'].max()}")
    print(f"気象データ数: {len(weather_df)}")
    
    # データの統合
    print("データを統合中...")
    merged_df = pd.merge(pe_df, weather_df, on='date', how='inner', suffixes=('_pe', '_weather'))
    
    print(f"統合後のデータ数: {len(merged_df)}")
    print(f"統合後の期間: {merged_df['date'].min()} から {merged_df['date'].max()}")
    
    # 基本統計
    print("\n=== 基本統計 ===")
    print(f"PE発生総数: {merged_df['people_pe'].sum()}")
    print(f"平均PE発生数/日: {merged_df['people_pe'].mean():.2f}")
    print(f"最大PE発生数/日: {merged_df['people_pe'].max()}")
    print(f"最小PE発生数/日: {merged_df['people_pe'].min()}")
    
    # 欠損値の確認
    print("\n=== 欠損値確認 ===")
    missing_pe = merged_df['people_pe'].isnull().sum()
    missing_weather = merged_df[['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].isnull().sum()
    print(f"PEデータ欠損: {missing_pe}")
    print(f"気象データ欠損:")
    for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
        print(f"  {col}: {missing_weather[col]}")
    
    # データの保存
    print("\n統合データを保存中...")
    merged_df.to_csv('pe_weather_merged.csv', index=False)
    print("統合データを保存しました: pe_weather_merged.csv")
    
    # サンプルデータの表示
    print("\n=== サンプルデータ ===")
    print(merged_df[['date', 'people_pe', 'avg_temp_weather', 'avg_humidity_weather', 'pressure_local']].head())
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_pe_weather_data()
    print("\n=== 統合完了 ===") 