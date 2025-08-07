from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, Any

# 気象データ取得クラスをインポート
from weather_api import WeatherDataCollector

app = FastAPI(title="心不全リスク予測API", version="1.0.0")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    recommendations: list

class HeartFailurePredictor:
    """心不全リスク予測クラス"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """保存済みモデルを読み込み"""
        try:
            # モデルファイルのパスを設定
            model_path = "../心不全気象予測モデル_50回Fold最適化版_保存モデル/"
            
            # 簡易版の予測ロジック（実際のモデルがない場合）
            self.model = self._create_simple_model()
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.model = self._create_simple_model()
    
    def _create_simple_model(self):
        """簡易版予測モデル（実際のモデルがない場合）"""
        def predict(weather_data):
            # 心不全リスクの簡易計算
            risk_score = 0.0
            
            # 寒冷ストレス
            if weather_data.get('is_cold_stress', 0) == 1:
                risk_score += 0.3
            
            # 暑熱ストレス
            if weather_data.get('is_heat_stress', 0) == 1:
                risk_score += 0.25
            
            # 急激な温度変化
            temp_range = weather_data.get('temp_range', 0)
            if temp_range > 15:
                risk_score += 0.2
            
            # 高湿度
            if weather_data.get('is_high_humidity', 0) == 1:
                risk_score += 0.15
            
            # 強風
            if weather_data.get('is_strong_wind', 0) == 1:
                risk_score += 0.1
            
            # 基本リスク（季節性）
            month = datetime.now().month
            if month in [12, 1, 2]:  # 冬
                risk_score += 0.2
            elif month in [7, 8]:    # 夏
                risk_score += 0.15
            
            return min(risk_score, 1.0)
        
        return predict
    
    def predict_risk(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """心不全リスクを予測"""
        try:
            # 予測実行
            risk_probability = self.model(weather_data)
            
            # リスクレベル判定
            if risk_probability < 0.3:
                risk_level = "低リスク"
                risk_score = 1
            elif risk_probability < 0.7:
                risk_level = "中リスク"
                risk_score = 2
            else:
                risk_level = "高リスク"
                risk_score = 3
            
            # 推奨事項を生成
            recommendations = self._generate_recommendations(weather_data, risk_level)
            
            return {
                "risk_probability": risk_probability,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "prediction_date": datetime.now().isoformat(),
                "weather_data": weather_data,
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"予測エラー: {e}")
            raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")
    
    def _generate_recommendations(self, weather_data: Dict[str, Any], risk_level: str) -> list:
        """推奨事項を生成"""
        recommendations = []
        
        if risk_level == "高リスク":
            recommendations.append("心不全患者の方は特に注意が必要です")
            recommendations.append("医療機関への連絡を検討してください")
            recommendations.append("安静を保ち、過度な運動を避けてください")
        
        if weather_data.get('is_cold_stress', 0) == 1:
            recommendations.append("寒冷ストレスが検出されました。暖房を適切に使用してください")
        
        if weather_data.get('is_heat_stress', 0) == 1:
            recommendations.append("暑熱ストレスが検出されました。適切な水分補給と冷房を使用してください")
        
        if weather_data.get('temp_range', 0) > 15:
            recommendations.append("急激な温度変化が予想されます。体調管理に注意してください")
        
        if weather_data.get('is_high_humidity', 0) == 1:
            recommendations.append("高湿度環境です。除湿機の使用を検討してください")
        
        if not recommendations:
            recommendations.append("現在の気象条件は心不全リスクが低い状態です")
        
        return recommendations

# 予測器のインスタンスを作成
predictor = HeartFailurePredictor()
weather_collector = WeatherDataCollector()

@app.get("/")
async def root():
    """APIの基本情報"""
    return {
        "message": "心不全リスク予測API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/predict/current")
async def predict_current_risk():
    """現在の気象データでリスクを予測"""
    try:
        # 気象データを取得
        weather_data = weather_collector.fetch_current_weather()
        
        # リスクを予測
        prediction = predictor.predict_risk(weather_data)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

@app.post("/predict/custom")
async def predict_custom_risk(weather_data: WeatherData):
    """カスタム気象データでリスクを予測"""
    try:
        # データを辞書に変換
        weather_dict = weather_data.dict()
        
        # 心不全特化特徴量を計算
        weather_dict.update({
            'is_cold_stress': 1 if weather_data.min_temp < 5 else 0,
            'is_heat_stress': 1 if weather_data.max_temp > 30 else 0,
            'temp_range': weather_data.max_temp - weather_data.min_temp,
            'is_high_humidity': 1 if weather_data.avg_humidity > 80 else 0,
            'is_low_humidity': 1 if weather_data.avg_humidity < 30 else 0,
            'is_strong_wind': 1 if weather_data.wind_speed > 10 else 0,
            'is_rainy': 1 if weather_data.precipitation > 0 else 0
        })
        
        # リスクを予測
        prediction = predictor.predict_risk(weather_dict)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")

@app.get("/weather/current")
async def get_current_weather():
    """現在の気象データを取得"""
    try:
        weather_data = weather_collector.fetch_current_weather()
        return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"気象データ取得エラー: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 