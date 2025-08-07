# AMI気象予測モデル 10年分最適化版 結果ファイル

## 概要
気象情報のみを使用した急性心筋梗塞（AMI）予測モデルの結果ファイルです。
10年分のデータを効果的に活用し、AUC 0.8954の高性能を達成しました。

## ファイル構成

### 性能評価ファイル
- **detailed_metrics.json**: 詳細な性能指標（JSON形式）
- **detailed_metrics.md**: 詳細な性能指標（Markdown形式）
- **detailed_results.json**: 詳細な結果データ
- **modelwise_metrics.json**: 各モデル別の性能指標
- **modelwise_metrics.md**: 各モデル別の性能指標（Markdown形式）

### 可視化ファイル
- **feature_importance.png**: 特徴量重要度のグラフ
- **feature_importance.csv**: 特徴量重要度のデータ
- **performance_curves_fold_1.png**: 1分割目の性能曲線
- **performance_curves_fold_2.png**: 2分割目の性能曲線
- **performance_curves_fold_3.png**: 3分割目の性能曲線

## 主要性能指標
- **AUC**: 0.8954 ± 0.0353
- **PR-AUC**: 0.8867 ± 0.0557
- **Precision**: 0.5169 ± 0.3719
- **Recall**: 0.5017 ± 0.4438
- **F1-score**: 0.4320 ± 0.3315

## 特徴
- 気象情報のみで90%のAUCを達成
- 20分割の時系列交差検証
- 600日分のテストデータ
- 64個の最適化された特徴量
- アンサンブル学習による高精度予測

## 作成日
2024年12月

## データ期間
2012年4月1日 - 2021年12月31日 