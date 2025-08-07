import pandas as pd
import numpy as np
import os

def classify_weather_condition(weather_text):
    """
    天気概況を日本語カテゴリに分類する関数
    
    分類カテゴリ:
    1. 快晴: 快晴
    2. 晴れ: 晴, 晴一時曇, 晴時々曇, 晴後曇, 晴後一時曇
    3. 薄曇: 薄曇, 薄曇時々晴, 薄曇一時晴
    4. 曇り: 曇, 曇一時晴, 曇時々晴, 曇後晴, 曇後一時晴
    5. 小雨: 雨, 雨一時曇, 雨時々曇, 雨後曇
    6. 大雨: 大雨, 大雨後曇, 大雨一時曇
    7. 雪: 雪, みぞれ, あられ
    8. 雷雨: 雷を伴う
    """
    if pd.isna(weather_text):
        return "欠損値"
    
    weather_text = str(weather_text).strip()
    
    # 1. 快晴
    if '快晴' in weather_text:
        return "快晴"
    
    # 2. 晴れ（晴が含まれ、曇・雨・雪・雷が含まれない）
    elif '晴' in weather_text and '曇' not in weather_text and '雨' not in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return "晴れ"
    
    # 3. 薄曇
    elif '薄曇' in weather_text:
        return "薄曇"
    
    # 4. 曇り（曇が含まれ、雨・雪・雷が含まれない）
    elif '曇' in weather_text and '雨' not in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return "曇り"
    
    # 6. 大雨
    elif '大雨' in weather_text:
        return "大雨"
    
    # 5. 小雨（雨が含まれ、雪・雷が含まれない）
    elif '雨' in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return "小雨"
    
    # 7. 雪
    elif '雪' in weather_text or 'みぞれ' in weather_text or 'あられ' in weather_text:
        return "雪"
    
    # 8. 雷雨
    elif '雷' in weather_text:
        return "雷雨"
    
    else:
        return "その他"

def classify_weather_data():
    print("天気概況データの日本語分類を開始します...")
    
    # ファイルパス
    input_path = "東京AMI天気データとJROAD結合後2012年4月1日〜2021年12月31日.csv"
    output_path = "weather_jroad_classified_2012_2021.csv"
    
    # データを読み込み
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"元データ行数: {len(df)}")
    print(f"元データ列数: {len(df.columns)}")
    
    # 天気概況の分類
    print("\n天気概況を分類中...")
    df['天気分類(昼)'] = df['天気概況(昼：06時～18時)'].apply(classify_weather_condition)
    df['天気分類(夜)'] = df['天気概況(夜：18時～翌日06時)'].apply(classify_weather_condition)
    
    # 分類結果の確認
    print("\n=== 昼間の天気概況分類結果 ===")
    day_counts = df['天気分類(昼)'].value_counts()
    for category, count in day_counts.items():
        print(f"{category}: {count}件")
    
    print("\n=== 夜間の天気概況分類結果 ===")
    night_counts = df['天気分類(夜)'].value_counts()
    for category, count in night_counts.items():
        print(f"{category}: {count}件")
    
    # 分類前後の比較
    print("\n=== 分類前後の比較（昼間・上位10件） ===")
    comparison_df = df[['天気概況(昼：06時～18時)', '天気分類(昼)']].head(10)
    for _, row in comparison_df.iterrows():
        original = row['天気概況(昼：06時～18時)']
        classified = row['天気分類(昼)']
        print(f"元: {original} → 分類: {classified}")
    
    print("\n=== 分類前後の比較（夜間・上位10件） ===")
    comparison_df = df[['天気概況(夜：18時～翌日06時)', '天気分類(夜)']].head(10)
    for _, row in comparison_df.iterrows():
        original = row['天気概況(夜：18時～翌日06時)']
        classified = row['天気分類(夜)']
        print(f"元: {original} → 分類: {classified}")
    
    # 欠損値の確認
    print(f"\n=== 欠損値の確認 ===")
    missing_day = df['天気分類(昼)'].isna().sum()
    missing_night = df['天気分類(夜)'].isna().sum()
    print(f"昼間の欠損値: {missing_day}件")
    print(f"夜間の欠損値: {missing_night}件")
    
    # 分類されたデータを保存
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n分類されたデータを {output_path} に保存しました。")
    
    print(f"\n=== 最終データ情報 ===")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    classify_weather_data() 