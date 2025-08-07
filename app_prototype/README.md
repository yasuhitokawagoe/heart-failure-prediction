# 心不全リスク予測アプリ

東京の気象データを自動取得して、心不全リスクをリアルタイムで予測するWebアプリケーションです。

## 🚀 機能

- **リアルタイム気象データ取得**: 気象庁APIから自動取得
- **心不全リスク予測**: 保存済みモデルを使用
- **リスクレベル表示**: 視覚的に分かりやすい表示
- **推奨事項**: リスクに応じた具体的なアドバイス
- **レスポンシブデザイン**: モバイル・デスクトップ対応

## 📁 ファイル構成

```
app_prototype/
├── weather_api.py          # 気象データ取得API
├── prediction_api.py       # 予測API（FastAPI）
├── requirements.txt        # Python依存関係
├── frontend/
│   └── index.html         # Webフロントエンド
└── README.md              # このファイル
```

## 🛠️ セットアップ

### 1. 依存関係のインストール

```bash
cd app_prototype
pip install -r requirements.txt
```

### 2. APIサーバーの起動

```bash
python prediction_api.py
```

サーバーは `http://localhost:8000` で起動します。

### 3. フロントエンドの起動

ブラウザで `frontend/index.html` を開くか、簡単なHTTPサーバーを起動：

```bash
cd frontend
python -m http.server 3000
```

フロントエンドは `http://localhost:3000` でアクセスできます。

## 📊 API エンドポイント

### 基本情報
- `GET /`: APIの基本情報
- `GET /health`: ヘルスチェック

### 予測関連
- `GET /predict/current`: 現在の気象データでリスク予測
- `POST /predict/custom`: カスタム気象データでリスク予測
- `GET /weather/current`: 現在の気象データ取得

## 🎯 使用方法

1. **APIサーバーを起動**
   ```bash
   python prediction_api.py
   ```

2. **ブラウザでフロントエンドを開く**
   - `frontend/index.html` を直接開く
   - または `http://localhost:3000` にアクセス

3. **リスク予測の確認**
   - ページが読み込まれると自動的に最新の気象データを取得
   - リスクレベル（低・中・高）と確率が表示される
   - 現在の気象状況と推奨事項も表示される

## 🔧 カスタマイズ

### 気象データ取得のカスタマイズ

`weather_api.py` の `WeatherDataCollector` クラスを編集：

```python
def fetch_current_weather(self):
    # カスタム気象データ取得ロジック
    pass
```

### 予測ロジックのカスタマイズ

`prediction_api.py` の `HeartFailurePredictor` クラスを編集：

```python
def _create_simple_model(self):
    # カスタム予測ロジック
    pass
```

### フロントエンドのカスタマイズ

`frontend/index.html` のCSSとJavaScriptを編集：

```css
/* カスタムスタイル */
.risk-level {
    /* カスタムデザイン */
}
```

## 🚀 本格的なデプロイ

### Docker化

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### クラウドデプロイ

1. **AWS/GCP/Azure** にデプロイ
2. **GitHub Actions** でCI/CD
3. **監視・ログ** の設定

## 🔮 今後の拡張

### 機能拡張
- [ ] 複数地域対応
- [ ] 他の疾患（AMI、脳卒中）の追加
- [ ] モバイルアプリ版
- [ ] プッシュ通知

### 技術拡張
- [ ] 実際の保存済みモデルの統合
- [ ] リアルタイム予測の精度向上
- [ ] 大規模データ処理
- [ ] AI/MLパイプライン

## 📝 注意事項

- このプロトタイプは簡易版の予測ロジックを使用しています
- 実際の医療用途には、より厳密な検証が必要です
- 気象庁APIの利用規約を遵守してください

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**開発者**: 心不全リスク予測アプリ開発チーム  
**バージョン**: 1.0.0  
**最終更新**: 2024年12月 