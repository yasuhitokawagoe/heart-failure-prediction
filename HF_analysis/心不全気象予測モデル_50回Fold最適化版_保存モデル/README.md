# 心不全気象予測モデル 50回Fold最適化版

## 概要
心不全患者の気象条件による悪化リスクを予測する機械学習モデルです。

## モデル性能
- **最良Fold**: 10
- **AUC**: 1.0000
- **PR-AUC**: 1.0000
- **F1最適化スコア**: 1.0000
- **保存日**: 2025-08-04T21:17:19.370505

## ファイル構成
- `model_info.json`: モデル情報と性能指標
- `best_hyperparameters.pkl`: 最良のハイパーパラメータ
- `ensemble_weights.npy`: アンサンブル重み
- `prediction_pipeline.py`: 予測用パイプライン

## 使用方法
```python
from prediction_pipeline import HeartFailureWeatherPredictor

# モデルを読み込み
predictor = HeartFailureWeatherPredictor("保存ディレクトリパス")

# 気象データを準備
weather_data = pd.DataFrame({
    'date': pd.date_range('2025-08-03', periods=7),
    'min_temp_weather': [20, 18, 15, 12, 10, 8, 5],
    'max_temp_weather': [30, 28, 25, 22, 20, 18, 15],
    'avg_temp_weather': [25, 23, 20, 17, 15, 13, 10],
    'avg_humidity_weather': [70, 75, 80, 85, 90, 85, 80],
    'pressure_local': [1013, 1010, 1008, 1005, 1002, 1000, 998],
    'avg_wind_weather': [5, 8, 10, 12, 15, 18, 20],
    'sunshine_hours_weather': [8, 6, 4, 2, 1, 0, 0],
    'precipitation': [0, 5, 10, 15, 20, 25, 30]
})

# 予測実行
predictions = predictor.predict(weather_data)
```

## 入力データ形式
必須の気象変数:
- date: 日付
- min_temp_weather: 最低気温
- max_temp_weather: 最高気温
- avg_temp_weather: 平均気温
- avg_humidity_weather: 平均湿度
- pressure_local: 現地気圧
- avg_wind_weather: 平均風速
- sunshine_hours_weather: 日照時間
- precipitation: 降水量

## 出力
- 予測確率 (0-1): 心不全悪化リスクの確率
- リスクレベル: 低リスク/中リスク/高リスク

## 注意事項
- モデルは東京のデータで学習されています
- 他の地域に適用する場合は再学習が必要です
- 気象データの欠損値は自動的に補完されます
