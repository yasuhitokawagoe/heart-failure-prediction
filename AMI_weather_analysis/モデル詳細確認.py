import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

def check_model_details():
    """保存されたモデルの詳細情報を確認"""
    try:
        # モデルの読み込み
        model = joblib.load('保存モデル/AMI予測モデル_天気概況6分類版_XGBoost.pkl')
        
        print("=== 保存されたモデルの詳細情報 ===")
        
        # モデルの基本情報
        print(f"モデルタイプ: {type(model)}")
        print(f"モデルパラメータ: {model.get_params()}")
        
        # モデルの内部構造を確認
        if hasattr(model, 'booster'):
            print(f"\nBooster情報:")
            print(f"  - 特徴量数: {model.booster.num_features()}")
            print(f"  - 木の数: {model.booster.num_boosted_rounds()}")
            
            # 特徴量名を取得してみる
            try:
                feature_names = model.booster.feature_names
                if feature_names:
                    print(f"  - 特徴量名: {len(feature_names)}個")
                    for i, name in enumerate(feature_names[:10]):  # 最初の10個のみ表示
                        print(f"    {i+1}: {name}")
                    if len(feature_names) > 10:
                        print(f"    ... 他 {len(feature_names)-10}個")
                else:
                    print("  - 特徴量名: 保存されていません")
            except Exception as e:
                print(f"  - 特徴量名取得エラー: {e}")
        
        # モデルの予測能力をテスト
        print(f"\n予測テスト:")
        try:
            # ダミーデータでテスト
            dummy_data = np.random.rand(10, 60)  # 60特徴量でテスト
            predictions = model.predict_proba(dummy_data)
            print(f"  - 予測成功: {predictions.shape}")
        except Exception as e:
            print(f"  - 予測エラー: {e}")
        
        return model
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

if __name__ == "__main__":
    check_model_details() 