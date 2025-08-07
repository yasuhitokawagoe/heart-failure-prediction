import pandas as pd
import numpy as np
import os

def encode_weather_condition(weather_text):
    if pd.isna(weather_text):
        return -1  # 欠損値
    weather_text = str(weather_text).strip()
    if '快晴' in weather_text:
        return 1  # 快晴
    elif '晴' in weather_text and '曇' not in weather_text and '雨' not in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return 2  # 晴れ
    elif '薄曇' in weather_text:
        return 3  # 薄曇
    elif '曇' in weather_text and '雨' not in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return 4  # 曇り
    elif '大雨' in weather_text:
        return 6  # 大雨
    elif '雨' in weather_text and '雪' not in weather_text and '雷' not in weather_text:
        return 5  # 小雨
    elif '雪' in weather_text or 'みぞれ' in weather_text or 'あられ' in weather_text:
        return 7  # 雪
    elif '雷' in weather_text:
        return 8  # 雷雨
    else:
        return 0  # その他（未分類）

def encode_weather_data():
    print("天気概況データのエンコードを開始します...")
    # ファイルパス
    input_path = "東京AMI天気データとJROAD結合後2012年4月1日〜2021年12月31日.csv"
    output_path = "weather_jroad_encoded_2012_2021.csv"
    
    # データを読み込み
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"元データ行数: {len(df)}")
    print(f"元データ列数: {len(df.columns)}")
    # 天気概況のエンコード
    print("\n天気概況をエンコード中...")
    df['weather_day_encoded'] = df['天気概況(昼：06時～18時)'].apply(encode_weather_condition)
    df['weather_night_encoded'] = df['天気概況(夜：18時～翌日06時)'].apply(encode_weather_condition)
    # エンコード結果の確認
    print("\n=== 昼間の天気概況エンコード結果 ===")
    day_counts = df['weather_day_encoded'].value_counts().sort_index()
    weather_labels = {0: 'その他', 1: '快晴', 2: '晴れ', 3: '薄曇', 4: '曇り', 5: '小雨', 6: '大雨', 7: '雪', 8: '雷雨', -1: '欠損値'}
    for code, count in day_counts.items():
        label = weather_labels.get(code, f'コード{code}')
        print(f"{label}: {count}件")
    print("\n=== 夜間の天気概況エンコード結果 ===")
    night_counts = df['weather_night_encoded'].value_counts().sort_index()
    for code, count in night_counts.items():
        label = weather_labels.get(code, f'コード{code}')
        print(f"{label}: {count}件")
    # エンコード前後の比較
    print("\n=== エンコード前後の比較（昼間・上位10件） ===")
    comparison_df = df[['天気概況(昼：06時～18時)', 'weather_day_encoded']].head(10)
    for _, row in comparison_df.iterrows():
        original = row['天気概況(昼：06時～18時)']
        encoded = row['weather_day_encoded']
        label = weather_labels.get(encoded, f'コード{encoded}')
        print(f"元: {original} → エンコード: {encoded} ({label})")
    # 欠損値の確認
    print(f"\n=== 欠損値の確認 ===")
    missing_day = df['weather_day_encoded'].isna().sum()
    missing_night = df['weather_night_encoded'].isna().sum()
    print(f"昼間の欠損値: {missing_day}件")
    print(f"夜間の欠損値: {missing_night}件")
    # 元の日本語列を削除（オプション）
    df_encoded = df.drop(['天気概況(昼：06時～18時)', '天気概況(夜：18時～翌日06時)'], axis=1)
    # エンコードされたデータを保存
    df_encoded.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nエンコードされたデータを {output_path} に保存しました。")
    print(f"\n=== 最終データ情報 ===")
    print(f"行数: {len(df_encoded)}")
    print(f"列数: {len(df_encoded.columns)}")
    print(f"列名: {list(df_encoded.columns)}")
    return df_encoded

if __name__ == "__main__":
    encode_weather_data() 