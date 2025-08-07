import pandas as pd
import numpy as np
import os

def get_weather_priority(weather_category):
    """
    天気カテゴリの優先度を取得する関数
    数値が大きいほど優先度が高い（悪い天気）
    """
    priority_map = {
        "雷雨": 8,
        "大雨": 7,
        "雪": 6,
        "小雨": 5,
        "曇り": 4,
        "薄曇": 3,
        "晴れ": 2,
        "快晴": 1,
        "欠損値": 0,
        "その他": 0
    }
    return priority_map.get(weather_category, 0)

def merge_weather_classifications(day_weather, night_weather):
    """
    昼と夜の天気分類を優先度ベースで統合する関数
    最も悪い天気（優先度が高い）を採用
    """
    if pd.isna(day_weather) and pd.isna(night_weather):
        return "欠損値"
    
    if pd.isna(day_weather):
        return night_weather
    if pd.isna(night_weather):
        return day_weather
    
    day_priority = get_weather_priority(day_weather)
    night_priority = get_weather_priority(night_weather)
    
    # 優先度が高い方（悪い天気）を採用
    if day_priority >= night_priority:
        return day_weather
    else:
        return night_weather

def merge_weather_data():
    print("昼と夜の天気分類を統合します...")
    
    # ファイルパス
    input_path = "weather_jroad_classified_2012_2021.csv"
    output_path = "weather_jroad_merged_2012_2021.csv"
    
    # データを読み込み
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"元データ行数: {len(df)}")
    print(f"元データ列数: {len(df.columns)}")
    
    # 天気分類の統合
    print("\n天気分類を統合中...")
    df['天気分類(統合)'] = df.apply(
        lambda row: merge_weather_classifications(
            row['天気分類(昼)'], 
            row['天気分類(夜)']
        ), 
        axis=1
    )
    
    # 統合結果の確認
    print("\n=== 統合後の天気分類結果 ===")
    merged_counts = df['天気分類(統合)'].value_counts()
    for category, count in merged_counts.items():
        print(f"{category}: {count}件")
    
    # 統合前後の比較
    print("\n=== 統合前後の比較（上位10件） ===")
    comparison_df = df[['天気分類(昼)', '天気分類(夜)', '天気分類(統合)']].head(10)
    for _, row in comparison_df.iterrows():
        day = row['天気分類(昼)']
        night = row['天気分類(夜)']
        merged = row['天気分類(統合)']
        print(f"昼: {day} | 夜: {night} → 統合: {merged}")
    
    # 優先度の確認
    print("\n=== 優先度マッピング ===")
    priority_map = {
        "雷雨": 8, "大雨": 7, "雪": 6, "小雨": 5, 
        "曇り": 4, "薄曇": 3, "晴れ": 2, "快晴": 1
    }
    for category, priority in sorted(priority_map.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: 優先度 {priority}")
    
    # 統合による変化の分析
    print("\n=== 統合による変化の分析 ===")
    changed_count = 0
    for _, row in df.iterrows():
        day = row['天気分類(昼)']
        night = row['天気分類(夜)']
        merged = row['天気分類(統合)']
        if day != merged and night != merged:
            changed_count += 1
    
    print(f"昼と夜の両方と異なる統合結果: {changed_count}件")
    
    # 昼と夜が異なる場合の分析
    different_count = 0
    for _, row in df.iterrows():
        day = row['天気分類(昼)']
        night = row['天気分類(夜)']
        if day != night:
            different_count += 1
    
    print(f"昼と夜が異なる日: {different_count}件")
    print(f"昼と夜が異なる割合: {different_count/len(df)*100:.1f}%")
    
    # 統合されたデータを保存
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n統合されたデータを {output_path} に保存しました。")
    
    print(f"\n=== 最終データ情報 ===")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    merge_weather_data() 