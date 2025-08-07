#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欠損値の詳細確認スクリプト
"""

import pandas as pd
import numpy as np

def check_missing_values():
    """欠損値の詳細確認"""
    print("🔍 欠損値の詳細確認開始")
    
    # データ読み込み
    file_path = '../東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv'
    
    print("📊 データ読み込み中...")
    df = pd.read_csv(file_path)
    
    # オリジナル期間（2012-2019年）に絞り込み
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= '2012-04-01') & (df['date'] <= '2019-12-31')]
    
    # カラム名を統一（people → hospitalization_count）
    if 'people' in df.columns:
        df['hospitalization_count'] = df['people']
    
    print(f"データ形状: {df.shape}")
    print(f"期間: {df['date'].min()} から {df['date'].max()}")
    
    # 基本情報
    print("\n=== 基本情報 ===")
    print(f"総行数: {len(df)}")
    print(f"総列数: {len(df.columns)}")
    
    # データ型確認
    print("\n=== データ型確認 ===")
    print(df.dtypes)
    
    # 欠損値確認
    print("\n=== 欠損値確認 ===")
    missing_count = df.isnull().sum()
    missing_ratio = (missing_count / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'missing_count': missing_count,
        'missing_ratio(%)': missing_ratio
    })
    
    # 欠損があるカラムのみ表示
    missing_columns = missing_info[missing_info['missing_count'] > 0]
    if len(missing_columns) > 0:
        print("欠損があるカラム:")
        print(missing_columns.sort_values('missing_count', ascending=False))
    else:
        print("✅ 欠損値はありません")
    
    # 数値カラムの確認
    print("\n=== 数値カラム確認 ===")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"数値カラム数: {len(numeric_columns)}")
    print(f"数値カラム: {numeric_columns}")
    
    # 文字列カラムの確認
    print("\n=== 文字列カラム確認 ===")
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"文字列カラム数: {len(string_columns)}")
    print(f"文字列カラム: {string_columns}")
    
    # 数値カラムの欠損確認
    print("\n=== 数値カラムの欠損確認 ===")
    numeric_missing = df[numeric_columns].isnull().sum()
    numeric_missing_ratio = (numeric_missing / len(df)) * 100
    
    numeric_missing_info = pd.DataFrame({
        'missing_count': numeric_missing,
        'missing_ratio(%)': numeric_missing_ratio
    })
    
    numeric_missing_columns = numeric_missing_info[numeric_missing_info['missing_count'] > 0]
    if len(numeric_missing_columns) > 0:
        print("数値カラムで欠損があるもの:")
        print(numeric_missing_columns.sort_values('missing_count', ascending=False))
    else:
        print("✅ 数値カラムに欠損値はありません")
    
    # サンプルデータ確認
    print("\n=== サンプルデータ確認 ===")
    print("最初の5行:")
    print(df.head())
    
    print("\n=== 数値カラムのサンプル ===")
    if len(numeric_columns) > 0:
        print(df[numeric_columns[:5]].head())
    
    # 異常値チェック
    print("\n=== 異常値チェック ===")
    for col in numeric_columns[:5]:  # 最初の5つの数値カラムのみ
        print(f"\n{col}:")
        print(f"  最小値: {df[col].min()}")
        print(f"  最大値: {df[col].max()}")
        print(f"  平均値: {df[col].mean():.2f}")
        print(f"  標準偏差: {df[col].std():.2f}")
        print(f"  サンプル値: {df[col].head(3).tolist()}")

if __name__ == "__main__":
    check_missing_values() 