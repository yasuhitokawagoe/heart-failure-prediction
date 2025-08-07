import joblib
import pandas as pd
import numpy as np

def check_saved_model_features():
    """保存されたモデルの特徴量を確認"""
    try:
        # モデルの読み込み
        model = joblib.load('保存モデル/AMI予測モデル_天気概況6分類版_XGBoost.pkl')
        
        print("=== 保存されたモデルの情報 ===")
        
        # モデルの特徴量名を取得
        if hasattr(model, 'feature_names_in_'):
            print(f"特徴量数: {len(model.feature_names_in_)}")
            print("特徴量名:")
            for i, name in enumerate(model.feature_names_in_):
                print(f"  {i+1}: {name}")
        else:
            print("モデルに特徴量名が保存されていません")
            
        # モデルの基本情報
        print(f"\nモデルタイプ: {type(model)}")
        print(f"モデルパラメータ: {model.get_params()}")
        
        return model
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

if __name__ == "__main__":
    check_saved_model_features() 