import shutil
import os

# コピー元とコピー先のパス
source_file = "AMI_weather_analysis/train_model_weather_only.py"
target_file = "AMI_weather_analysis/train_model_weather_only_with_save.py"

try:
    # ファイルの存在確認
    if os.path.exists(source_file):
        print(f"✓ コピー元ファイル存在確認: {source_file}")
        
        # コピー実行
        shutil.copy2(source_file, target_file)
        print(f"✓ コピー完了: {target_file}")
        
        # コピー結果確認
        if os.path.exists(target_file):
            print(f"✓ コピー先ファイル確認: {target_file}")
            print("原本は保持されました。")
        else:
            print("❌ コピー先ファイルが見つかりません")
            
    else:
        print(f"❌ コピー元ファイルが見つかりません: {source_file}")
        
except Exception as e:
    print(f"❌ コピー中にエラーが発生しました: {e}") 