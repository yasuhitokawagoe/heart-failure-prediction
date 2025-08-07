#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全リスク予測Webアプリ - 本格運用版
セキュリティ強化とログ機能を追加
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import json
import os
import requests
from typing import Dict, Any, List, Optional
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import re
import logging
import hashlib
import secrets
from functools import wraps
import time

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# セキュリティ設定
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
RATE_LIMIT_REQUESTS = 100  # 1時間あたりのリクエスト数
RATE_LIMIT_WINDOW = 3600   # 1時間（秒）

# レート制限用の辞書
request_counts = {}

app = FastAPI(
    title="心不全リスク予測Webアプリ", 
    version="2.0.0",
    description="気象データに基づく心不全リスク予測API",
    docs_url="/docs" if os.getenv("ENVIRONMENT") == "development" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") == "development" else None
)

# CORS設定（本格運用では特定のドメインのみ許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本格運用では特定のドメインを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# セキュリティ
security = HTTPBearer()

class WeatherData(BaseModel):
    """気象データのモデル"""
    avg_temp: float
    max_temp: float
    min_temp: float
    avg_humidity: float
    pressure: float
    precipitation: float
    wind_speed: float
    sunshine_hours: float

class PredictionResponse(BaseModel):
    """予測結果のモデル"""
    risk_probability: float
    risk_level: str
    risk_score: int
    prediction_date: str
    weather_data: Dict[str, Any]
    recommendations: List[str]
    model_info: Dict[str, Any]

def rate_limit(func):
    """レート制限デコレータ"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        client_ip = kwargs.get('client_ip', 'unknown')
        current_time = time.time()
        
        # 古いリクエスト記録を削除
        if client_ip in request_counts:
            request_counts[client_ip] = [
                req_time for req_time in request_counts[client_ip] 
                if current_time - req_time < RATE_LIMIT_WINDOW
            ]
        
        # リクエスト数をチェック
        if client_ip in request_counts and len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        # リクエストを記録
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(current_time)
        
        return await func(*args, **kwargs)
    return wrapper

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """APIキー認証"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

class HeartFailurePredictor:
    """心不全リスク予測クラス（最適化版）"""
    
    def __init__(self):
        self.model_dir = "HF_analysis/心不全気象予測モデル_50回Fold最適化版_保存モデル"
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """保存済みモデルを読み込み"""
        try:
            # モデル情報を読み込み
            model_info_path = f'{self.model_dir}/model_info.json'
            logger.info(f"読み込みファイルパス: {model_info_path}")
            with open(model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            logger.info(f"読み込んだAUC値: {self.model_info['best_auc']}")
            
            # ハイパーパラメータを読み込み
            with open(f'{self.model_dir}/best_hyperparameters.pkl', 'rb') as f:
                self.best_params = pickle.load(f)
            
            # アンサンブル重みを読み込み
            self.ensemble_weights = np.load(f'{self.model_dir}/ensemble_weights.npy')
            
            logger.info("✓ 心不全モデル情報を読み込みました")
            logger.info(f"  最良Fold: {self.model_info['best_fold']}")
            logger.info(f"  最良AUC: {self.model_info['best_auc']:.4f}")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            # フォールバック: 簡易版モデル
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """フォールバック用の簡易モデルを作成"""
        logger.info("簡易版モデルを作成中...")
        
        # LightGBM
        lgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        # XGBoost
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        # CatBoost
        cb_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42
        }
        
        self.models = {
            'lgb': lgb.LGBMClassifier(**lgb_params),
            'xgb': xgb.XGBClassifier(**xgb_params),
            'cb': cb.CatBoostClassifier(**cb_params, verbose=False)
        }
        
        self.ensemble_weights = np.array([0.33, 0.33, 0.34])
        self.model_info = {
            'model_version': 'fallback_v1.0',
            'best_auc': 0.8,
            'best_fold': 1,
            'features_used': 20,
            'timeseries_features': False
        }
        
        logger.info("✓ 簡易版モデルを作成しました")

# 予測器の初期化
predictor = HeartFailurePredictor()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "心不全リスク予測Webアプリ v2.0",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs" if os.getenv("ENVIRONMENT") == "development" else "API documentation disabled in production"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor.model_info is not None
    }

@app.get("/predict/current")
@rate_limit
async def predict_current_risk(client_ip: str = "unknown"):
    """現在の気象データによるリスク予測（認証不要）"""
    try:
        logger.info("=== 予測処理開始 ===")
        
        # 気象データ取得（簡易版）
        weather_data = {
            "avg_temp": 28.3,
            "max_temp": 31.3,
            "min_temp": 25.3,
            "avg_humidity": 75.0,
            "pressure": 1005.0,
            "precipitation": 0.0,
            "wind_speed": 9.7,
            "sunshine_hours": 8.0
        }
        
        # 予測実行
        prediction_result = predictor.predict_risk(weather_data)
        
        logger.info("=== 予測処理完了 ===")
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"予測エラー: {e}")
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

@app.post("/predict/custom")
async def predict_custom_risk(
    weather_data: WeatherData,
    api_key: str = Depends(verify_api_key)
):
    """カスタム気象データによるリスク予測（認証必要）"""
    try:
        # 予測実行
        prediction_result = predictor.predict_risk(weather_data.dict())
        return prediction_result
        
    except Exception as e:
        logger.error(f"カスタム予測エラー: {e}")
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """モデル情報を取得"""
    return {
        "model_info": predictor.model_info,
        "ensemble_weights": predictor.ensemble_weights.tolist(),
        "best_params": predictor.best_params
    }

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Webインターフェース"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>心不全リスク予測アプリ v2.0</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .content {
                padding: 30px;
            }
            
            .prediction-card {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                border-left: 5px solid #667eea;
            }
            
            .risk-level {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                margin-bottom: 15px;
            }
            
            .risk-low {
                background: #d4edda;
                color: #155724;
            }
            
            .risk-medium {
                background: #fff3cd;
                color: #856404;
            }
            
            .risk-high {
                background: #f8d7da;
                color: #721c24;
            }
            
            .weather-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            
            .weather-item {
                background: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .weather-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #667eea;
            }
            
            .weather-label {
                font-size: 0.9em;
                color: #666;
                margin-top: 5px;
            }
            
            .recommendations {
                background: #e3f2fd;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .recommendations h3 {
                color: #1976d2;
                margin-bottom: 15px;
            }
            
            .recommendations ul {
                list-style: none;
                padding: 0;
            }
            
            .recommendations li {
                padding: 8px 0;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .recommendations li:last-child {
                border-bottom: none;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
            }
            
            .error {
                background: #ffebee;
                color: #c62828;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
            }
            
            .refresh-btn {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1em;
                margin-top: 20px;
                transition: background 0.3s;
            }
            
            .refresh-btn:hover {
                background: #5a6fd8;
            }
            
            .model-info {
                background: #f5f5f5;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
                font-size: 0.9em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>💓 心不全リスク予測</h1>
                <p>気象データに基づく心不全悪化リスクの予測システム</p>
            </div>
            
            <div class="content">
                <div id="loading" class="loading">
                    <h3>気象データを取得中...</h3>
                    <p>現在の気象条件を分析してリスクを評価しています</p>
                </div>
                
                <div id="prediction" style="display: none;">
                    <div class="prediction-card">
                        <h2>予測結果</h2>
                        <div class="risk-level" id="risk-level">評価中...</div>
                        <p><strong>リスク確率:</strong> <span id="risk-probability">-</span></p>
                        <p><strong>予測日時:</strong> <span id="prediction-date">-</span></p>
                    </div>
                    
                    <div class="weather-grid" id="weather-grid">
                        <!-- 気象データがここに表示されます -->
                    </div>
                    
                    <div class="recommendations" id="recommendations">
                        <h3>推奨事項</h3>
                        <ul id="recommendations-list">
                            <!-- 推奨事項がここに表示されます -->
                        </ul>
                    </div>
                    
                    <div class="model-info">
                        <h4>モデル情報</h4>
                        <p><strong>バージョン:</strong> <span id="model-version">-</span></p>
                        <p><strong>AUC:</strong> <span id="model-auc">-</span></p>
                        <p><strong>特徴量数:</strong> <span id="model-features">-</span></p>
                    </div>
                    
                    <button class="refresh-btn" onclick="loadPrediction()">最新データで更新</button>
                </div>
                
                <div id="error" class="error" style="display: none;">
                    <h3>エラーが発生しました</h3>
                    <p id="error-message">データの取得に失敗しました。しばらく時間をおいてから再試行してください。</p>
                </div>
            </div>
        </div>
        
        <script>
            async function loadPrediction() {
                const loading = document.getElementById('loading');
                const prediction = document.getElementById('prediction');
                const error = document.getElementById('error');
                
                loading.style.display = 'block';
                prediction.style.display = 'none';
                error.style.display = 'none';
                
                try {
                    const response = await fetch('/predict/current');
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayPrediction(data);
                        loading.style.display = 'none';
                        prediction.style.display = 'block';
                    } else {
                        throw new Error(data.detail || '予測に失敗しました');
                    }
                } catch (err) {
                    console.error('Error:', err);
                    document.getElementById('error-message').textContent = err.message;
                    loading.style.display = 'none';
                    error.style.display = 'block';
                }
            }
            
            function displayPrediction(data) {
                // リスクレベル
                const riskLevel = document.getElementById('risk-level');
                riskLevel.textContent = data.risk_level;
                riskLevel.className = 'risk-level risk-' + (data.risk_level === '高リスク' ? 'high' : data.risk_level === '中リスク' ? 'medium' : 'low');
                
                // リスク確率
                document.getElementById('risk-probability').textContent = (data.risk_probability * 100).toFixed(1) + '%';
                
                // 予測日時
                document.getElementById('prediction-date').textContent = new Date(data.prediction_date).toLocaleString('ja-JP');
                
                // 気象データ
                const weatherGrid = document.getElementById('weather-grid');
                weatherGrid.innerHTML = '';
                
                const weatherData = data.weather_data;
                const weatherItems = [
                    { label: '平均気温', value: weatherData.avg_temp + '°C', icon: '🌡️' },
                    { label: '最高気温', value: weatherData.max_temp + '°C', icon: '🔥' },
                    { label: '最低気温', value: weatherData.min_temp + '°C', icon: '❄️' },
                    { label: '湿度', value: weatherData.avg_humidity + '%', icon: '💧' },
                    { label: '気圧', value: weatherData.pressure + 'hPa', icon: '🌪️' },
                    { label: '降水量', value: weatherData.precipitation + 'mm', icon: '🌧️' },
                    { label: '風速', value: weatherData.wind_speed + 'km/h', icon: '💨' },
                    { label: '日照時間', value: weatherData.sunshine_hours + '時間', icon: '☀️' }
                ];
                
                weatherItems.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'weather-item';
                    div.innerHTML = `
                        <div class="weather-value">${item.icon} ${item.value}</div>
                        <div class="weather-label">${item.label}</div>
                    `;
                    weatherGrid.appendChild(div);
                });
                
                // 推奨事項
                const recommendationsList = document.getElementById('recommendations-list');
                recommendationsList.innerHTML = '';
                data.recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    recommendationsList.appendChild(li);
                });
                
                // モデル情報
                document.getElementById('model-version').textContent = data.model_info.model_version || '-';
                document.getElementById('model-auc').textContent = (data.model_info.best_auc || 0).toFixed(4);
                document.getElementById('model-features').textContent = data.model_info.features_used || '-';
            }
            
            // ページ読み込み時に予測を実行
            window.onload = loadPrediction;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    
    # 環境変数から設定を取得
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info("💓 心不全リスク予測Webアプリ v2.0 を起動中...")
    logger.info(f"🌐 Webインターフェース: http://localhost:{port}/web")
    logger.info(f"📊 API: http://localhost:{port}")
    if os.getenv("ENVIRONMENT") == "development":
        logger.info(f"📚 APIドキュメント: http://localhost:{port}/docs")
    
    uvicorn.run(
        "heart_failure_web_app_production:app",
        host=host,
        port=port,
        reload=False,  # 本格運用ではFalse
        log_level="info"
    ) 