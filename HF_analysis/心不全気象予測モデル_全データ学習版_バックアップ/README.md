# 心不全気象予測モデル 全データ学習版

## 📋 概要

このモデルは、心不全患者の入院リスクを気象データから予測する機械学習モデルです。2012年から2021年までの全データを使用して学習された、実運用向けの最終モデルです。

## 🎯 モデルの特徴

### 学習データ
- **期間**: 2012年4月1日 〜 2021年12月31日
- **データ数**: 3,562件
- **特徴量数**: 123個
- **高リスク日割合**: 25.0%

### 学習方法
- **Hold-out法**: 学習データ（2012-2020年）とテストデータ（2021年）に分割
- **アンサンブル学習**: LightGBM、XGBoost、CatBoostの3つのモデルを組み合わせ
- **ハイパーパラメータ最適化**: Optunaを使用した50回の試行

### 性能評価結果
- **AUC**: 0.9004
- **PR-AUC**: 0.8208
- **Precision**: 0.9012
- **Recall**: 0.6186
- **F1-Score**: 0.7337

## 📁 ファイル構成

### モデルファイル
- `lgb_model_final.pkl` - LightGBMモデル（578KB）
- `xgb_model_final.pkl` - XGBoostモデル（744KB）
- `cb_model_final.pkl` - CatBoostモデル（175KB）
- `ensemble_weights.npy` - アンサンブル重み
- `best_hyperparameters.pkl` - 最適化されたハイパーパラメータ
- `feature_columns.json` - 使用特徴量リスト
- `model_info.json` - モデル情報

### 学習コード
- `心不全気象予測モデル_全データ学習版.py` - 学習スクリプト

### 評価結果
- `心不全気象予測モデル_全データ学習版_結果/` - 詳細な評価結果

## 🔧 特徴量エンジニアリング

### 心不全特化の特徴量
1. **日付特徴量**
   - 季節性（冬期フラグ、夏期フラグなど）
   - 曜日パターン（月曜日、金曜日、週末）
   - 祝日・連休フラグ
   - 月末・月初フラグ

2. **気象特徴量**
   - 温度変化（日較差、前日比、3日前比）
   - ストレス指標（寒冷ストレス、暑熱ストレス、温度ショック）
   - 湿度関連（高湿度、低湿度、湿度変化）
   - 気圧関連（気圧変化、気圧低下・上昇）
   - 風・降水関連

3. **高度な特徴量**
   - 移動平均・標準偏差（3日、7日、14日）
   - 季節性を考慮した重み付け特徴量
   - 心不全悪化リスクの複合指標
   - 気象ストレスの累積効果
   - 急激な変化の検出（加速度）

4. **相互作用特徴量**
   - 温度×湿度の相互作用
   - 温度×気圧の相互作用
   - 季節×気象の相互作用
   - 曜日×気象の相互作用

## 🚀 使用方法

### モデルの読み込み
```python
import joblib
import numpy as np
import json

# モデルの読み込み
lgb_model = joblib.load('lgb_model_final.pkl')
xgb_model = joblib.load('xgb_model_final.pkl')
cb_model = joblib.load('cb_model_final.pkl')

# アンサンブル重みの読み込み
weights = np.load('ensemble_weights.npy')

# 特徴量リストの読み込み
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)
```

### 予測の実行
```python
# 特徴量の準備（123個の特徴量が必要）
features = prepare_features(weather_data)

# 各モデルでの予測
lgb_pred = lgb_model.predict_proba(features)[:, 1]
xgb_pred = xgb_model.predict_proba(features)[:, 1]
cb_pred = cb_model.predict_proba(features)[:, 1]

# アンサンブル予測
ensemble_pred = (weights[0] * lgb_pred + 
                weights[1] * xgb_pred + 
                weights[2] * cb_pred)

# リスク確率（0-1の値）
risk_probability = ensemble_pred[0]
```

## 📊 モデルの信頼性

### 評価方法
- **Hold-out法**: 学習データ（2012-2020年）で学習、テストデータ（2021年）で評価
- **実際の運用**: 全データ（2012-2021年）で最終モデルを作成

### 性能指標
- **AUC 0.9004**: 高い識別性能
- **Precision 0.9012**: 高い精度（偽陽性が少ない）
- **Recall 0.6186**: 適度な検出率
- **F1-Score 0.7337**: バランスの取れた性能

## ⚠️ 注意事項

1. **データの完全性**: 欠損値がある場合はエラーを返す（勝手な補完は行わない）
2. **特徴量の順序**: 特徴量は`feature_columns.json`で指定された順序で入力する必要がある
3. **データの前処理**: 標準化などの前処理は学習時に使用した方法と一致させる必要がある

## 📅 作成日時

- **学習完了日**: 2025年8月4日
- **総実行時間**: 5.4分
- **モデルバージョン**: 全データ学習版_v1.0

## 🔄 更新履歴

- **v1.0**: 初回リリース（全データ学習版）
  - Hold-out法による性能評価
  - アンサンブル学習（LightGBM + XGBoost + CatBoost）
  - 心不全特化の特徴量エンジニアリング

---

**作成者**: 心不全気象予測プロジェクト  
**用途**: 心不全患者の入院リスク予測  
**対象地域**: 東京（気象データに基づく） 