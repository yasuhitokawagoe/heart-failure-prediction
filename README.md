# 心不全リスク予測Webアプリ

## 📋 概要

このプロジェクトは、気象データを基に心不全リスクを予測するWebアプリケーションです。機械学習モデルを使用して、気温、湿度、気圧などの気象条件から心不全のリスクを評価し、適切な推奨事項を提供します。

## 🚀 主な機能

- **リアルタイム気象データ取得**: Open-Meteo APIを使用した現在の気象データ取得
- **心不全リスク予測**: 機械学習モデルによるリスク確率の計算
- **推奨事項生成**: リスクレベルに応じた個別化された推奨事項
- **Webインターフェース**: 直感的なWeb UIでの結果表示
- **RESTful API**: 外部システムとの連携可能なAPI

## 🛠️ 技術スタック

- **バックエンド**: FastAPI (Python)
- **機械学習**: LightGBM, XGBoost, CatBoost (アンサンブル)
- **フロントエンド**: HTML, CSS, JavaScript
- **データ取得**: Open-Meteo API, 気象庁API
- **モデル評価**: Hold-out validation (AUC: 0.9004)

## 📊 モデル情報

- **モデルバージョン**: 全データ学習版_v1.0
- **評価方法**: Hold-out validation
- **性能指標**: AUC = 0.9004
- **特徴量数**: 127個
- **学習データ**: 2012-2021年の気象・心不全データ

## 🚀 セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. アプリケーションの起動
```bash
python heart_failure_web_app.py
```

### 4. アクセス
- **Webインターフェース**: http://localhost:8000/web
- **API**: http://localhost:8000
- **APIドキュメント**: http://localhost:8000/docs

## 📁 プロジェクト構造

```
heart-failure-prediction/
├── heart_failure_web_app.py          # メインアプリケーション
├── HF_analysis/                      # 機械学習関連
│   ├── 心不全気象予測モデル_全データ学習版.py
│   ├── 心不全気象予測モデル_全データ学習版_保存モデル/
│   └── 心不全気象予測モデル_全データ学習版_バックアップ/
├── requirements.txt                   # 依存関係
├── README.md                         # このファイル
└── .gitignore                        # Git除外設定
```

## 🔧 API エンドポイント

### 予測関連
- `GET /predict/current` - 現在の気象データでのリスク予測
- `POST /predict/custom` - カスタム気象データでのリスク予測

### データ関連
- `GET /weather/current` - 現在の気象データ取得
- `GET /data/complete` - 完全な気象データ取得
- `GET /data/statistics` - 統計情報取得

### システム関連
- `GET /health` - ヘルスチェック
- `GET /model/info` - モデル情報取得
- `GET /cache/status` - キャッシュ状態確認

## 📈 予測結果の例

```json
{
  "risk_probability": 0.22,
  "risk_level": "低リスク",
  "risk_score": 1,
  "recommendations": {
    "recommendations": [
      "✅ 低リスク状態です。現在の体調管理を継続してください。",
      "📊 気象ストレスが蓄積しています。長期的な体調管理が必要です。",
      "💊 処方された薬は指示通りに服用してください。",
      "🏃‍♂️ 適度な運動と十分な休息を心がけてください。",
      "🥗 塩分制限とバランスの良い食事を心がけてください。"
    ]
  }
}
```

## 🤝 貢献

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は `LICENSE` ファイルを参照してください。

## 📞 お問い合わせ

- **作成者**: [Your Name]
- **メール**: [your-email@example.com]
- **プロジェクトURL**: https://github.com/your-username/heart-failure-prediction

## 🙏 謝辞

- 気象庁: 気象データの提供
- Open-Meteo: リアルタイム気象API
- 医療関係者の方々: 専門的なアドバイス

---

**⚠️ 免責事項**: このアプリケーションは研究・教育目的で作成されています。医療診断や治療の代わりとして使用しないでください。健康上の懸念がある場合は、必ず医療専門家に相談してください。 