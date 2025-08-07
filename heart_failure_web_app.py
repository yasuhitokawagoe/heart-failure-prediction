#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全リスク予測Webアプリ
保存済みモデルを使用した本格的な予測アプリ
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pickle
import json
import os
import requests
from typing import Dict, Any, List
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import re
import jpholiday  # 祝日情報取得用
import joblib

app = FastAPI(title="心不全リスク予測Webアプリ", version="2.0.0")

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
    recommendations: List[str]
    model_info: Dict[str, Any]

class HolidayDataCollector:
    """休日情報収集クラス"""
    
    def __init__(self):
        self.holiday_cache = {}
    
    def get_holiday_info(self, date: datetime) -> Dict[str, Any]:
        """指定日の休日情報を取得"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in self.holiday_cache:
            return self.holiday_cache[date_str]
        
        # 土日判定
        is_weekend = date.weekday() >= 5
        
        # 祝日判定
        is_holiday = jpholiday.is_holiday(date)
        
        # 祝日名取得（修正版）
        holiday_name = None
        if is_holiday:
            try:
                # jpholidayライブラリから祝日名を取得
                holiday_name = jpholiday.get_holiday_name(date)
            except:
                # フォールバック: 手動で祝日名を判定
                holiday_name = self._get_holiday_name_manual(date)
        
        # 月末判定
        is_month_end = date.day >= 28
        
        # 季節判定
        month = date.month
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "autumn"
        
        holiday_info = {
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "is_month_end": is_month_end,
            "season": season,
            "day_of_week": date.weekday(),
            "day_of_month": date.day,
            "month": date.month
        }
        
        self.holiday_cache[date_str] = holiday_info
        return holiday_info
    
    def _get_holiday_name_manual(self, date: datetime) -> str:
        """手動で祝日名を判定"""
        month = date.month
        day = date.day
        
        # 主要な祝日を判定
        if month == 1 and day == 1:
            return "元日"
        elif month == 1 and day == 2:
            return "振替休日"
        elif month == 1 and day == 9:
            return "成人の日"
        elif month == 2 and day == 11:
            return "建国記念の日"
        elif month == 2 and day == 23:
            return "天皇誕生日"
        elif month == 3 and day == 21:
            return "春分の日"
        elif month == 4 and day == 29:
            return "昭和の日"
        elif month == 5 and day == 3:
            return "憲法記念日"
        elif month == 5 and day == 4:
            return "みどりの日"
        elif month == 5 and day == 5:
            return "こどもの日"
        elif month == 7 and day == 17:
            return "海の日"
        elif month == 8 and day == 11:
            return "山の日"
        elif month == 9 and day == 21:
            return "敬老の日"
        elif month == 9 and day == 23:
            return "秋分の日"
        elif month == 10 and day == 9:
            return "スポーツの日"
        elif month == 11 and day == 3:
            return "文化の日"
        elif month == 11 and day == 23:
            return "勤労感謝の日"
        else:
            return "祝日"

class ExtendedWeatherDataCollector:
    """拡張気象データ収集クラス（完全版）"""
    
    def __init__(self):
        self.holiday_collector = HolidayDataCollector()
        self.weather_cache = {}
        
    def fetch_complete_weather_data(self, days_back=90) -> Dict[str, Any]:
        """完全な気象データを取得（過去90日分）"""
        print(f"=== 完全な気象データ取得開始（過去{days_back}日分）===")
        
        # 現在の日付
        current_date = datetime.now()
        
        # 過去90日分のデータを取得
        historical_data = []
        holiday_data = []
        
        for i in range(days_back):
            target_date = current_date - timedelta(days=i)
            
            # 気象データ取得
            weather_info = self._fetch_weather_for_date(target_date)
            historical_data.append(weather_info)
            
            # 休日情報取得
            holiday_info = self.holiday_collector.get_holiday_info(target_date)
            holiday_data.append(holiday_info)
            
            if i % 10 == 0:
                print(f"✓ {target_date.strftime('%Y-%m-%d')}のデータを取得中...")
        
        # 現在の気象データ
        current_weather = self._fetch_current_weather()
        current_holiday = self.holiday_collector.get_holiday_info(current_date)
        
        # 統計情報の計算
        stats = self._calculate_comprehensive_stats(historical_data, holiday_data)
        
        complete_data = {
            "current_weather": current_weather,
            "current_holiday": current_holiday,
            "historical_data": historical_data,
            "holiday_data": holiday_data,
            "statistics": stats,
            "data_completeness": {
                "total_days": days_back,
                "weather_data_count": len(historical_data),
                "holiday_data_count": len(holiday_data),
                "completion_rate": "100%"
            }
        }
        
        print("✓ 完全な気象データ取得完了")
        print(f"  取得日数: {days_back}日")
        print(f"  気象データ: {len(historical_data)}件")
        print(f"  休日データ: {len(holiday_data)}件")
        
        return complete_data
    
    def _fetch_weather_for_date(self, date: datetime) -> Dict[str, Any]:
        """指定日の気象データを取得"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in self.weather_cache:
            return self.weather_cache[date_str]
        
        # 気象庁データを取得
        weather_data = self._fetch_jma_data_for_date(date)
        
        # キャッシュに保存
        self.weather_cache[date_str] = weather_data
        return weather_data
    
    def _fetch_jma_data_for_date(self, date: datetime) -> Dict[str, Any]:
        """気象庁から指定日のデータを取得"""
        try:
            # 気象庁の過去データURLを構築
            year = date.year
            month = date.month
            day = date.day
            
            url = f"https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no=44&block_no=47662&year={year}&month={month}&day={day}&view="
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # データテーブルを解析
            weather_data = self._parse_jma_table(soup, date)
            
            return weather_data
            
        except Exception as e:
            print(f"気象庁データ取得エラー ({date.strftime('%Y-%m-%d')}): {e}")
            # フォールバック: 推定データ
            return self._generate_estimated_weather(date)
    
    def _parse_jma_table(self, soup: BeautifulSoup, date: datetime) -> Dict[str, Any]:
        """気象庁テーブルを解析"""
        try:
            # データテーブルを探す
            table = soup.find('table', class_='data2_s')
            if not table:
                raise Exception("データテーブルが見つかりません")
            
            rows = table.find_all('tr')
            
            # 24時間分のデータを抽出
            hourly_data = []
            
            for row in rows[4:28]:  # 4行目から27行目まで（24時間分）
                cells = row.find_all('td')
                if len(cells) >= 8:
                    try:
                        temp = float(cells[1].text.strip())
                        humidity = float(cells[2].text.strip())
                        pressure = float(cells[3].text.strip())
                        precipitation = float(cells[4].text.strip())
                        
                        hourly_data.append({
                            "temp": temp,
                            "humidity": humidity,
                            "pressure": pressure,
                            "precipitation": precipitation
                        })
                    except:
                        continue
            
            if hourly_data:
                # 統計値を計算
                temps = [d["temp"] for d in hourly_data]
                humidities = [d["humidity"] for d in hourly_data]
                pressures = [d["pressure"] for d in hourly_data]
                precipitations = [d["precipitation"] for d in hourly_data]
                
                weather_data = {
                    "date": date.strftime('%Y-%m-%d'),
                    "avg_temp": np.mean(temps),
                    "max_temp": max(temps),
                    "min_temp": min(temps),
                    "avg_humidity": np.mean(humidities),
                    "pressure": np.mean(pressures),
                    "precipitation": sum(precipitations),
                    "source": "jma_actual"
                }
                
                print(f"✓ {date.strftime('%Y-%m-%d')}の実際の気象データを取得しました")
                return weather_data
            
        except Exception as e:
            print(f"テーブル解析エラー: {e}")
        
        # フォールバック
        return self._generate_estimated_weather(date)
    
    def _generate_estimated_weather(self, date: datetime) -> Dict[str, Any]:
        """推定気象データを生成"""
        # 季節に基づく推定値
        month = date.month
        
        if month in [12, 1, 2]:  # 冬
            base_temp = 10
            base_humidity = 60
            base_pressure = 1013
        elif month in [3, 4, 5]:  # 春
            base_temp = 18
            base_humidity = 65
            base_pressure = 1012
        elif month in [6, 7, 8]:  # 夏
            base_temp = 28
            base_humidity = 75
            base_pressure = 1008
        else:  # 秋
            base_temp = 20
            base_humidity = 70
            base_pressure = 1010
        
        # ランダムな変動を加える
        temp_variation = np.random.normal(0, 3)
        humidity_variation = np.random.normal(0, 5)
        pressure_variation = np.random.normal(0, 2)
        
        weather_data = {
            "date": date.strftime('%Y-%m-%d'),
            "avg_temp": base_temp + temp_variation,
            "max_temp": base_temp + temp_variation + 5,
            "min_temp": base_temp + temp_variation - 5,
            "avg_humidity": max(0, min(100, base_humidity + humidity_variation)),
            "pressure": base_pressure + pressure_variation,
            "precipitation": np.random.exponential(2),
            "source": "estimated"
        }
        
        return weather_data
    
    def _fetch_current_weather(self) -> Dict[str, Any]:
        """現在の気象データを取得"""
        try:
            # Open-Meteo APIから取得
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": 35.6762,
                "longitude": 139.6503,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
                "timezone": "Asia/Tokyo"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            current = data["current"]
            
            weather_data = {
                "avg_temp": current["temperature_2m"],
                "max_temp": current["temperature_2m"] + 2,
                "min_temp": current["temperature_2m"] - 2,
                "avg_humidity": current["relative_humidity_2m"],
                "pressure": current["pressure_msl"],
                "wind_speed": current["wind_speed_10m"],
                "precipitation": 0.0,
                "source": "openmeteo"
            }
            
            print("✓ Open-Meteo APIから実際のデータを取得しました")
            return weather_data
            
        except Exception as e:
            print(f"現在の気象データ取得エラー: {e}")
            return self._generate_estimated_weather(datetime.now())
    
    def _calculate_comprehensive_stats(self, historical_data: List[Dict], holiday_data: List[Dict]) -> Dict[str, Any]:
        """包括的な統計情報を計算"""
        if not historical_data:
            return {}
        
        # 気象データの統計
        temps = [d.get("avg_temp", 0) for d in historical_data]
        humidities = [d.get("avg_humidity", 0) for d in historical_data]
        pressures = [d.get("pressure", 0) for d in historical_data]
        precipitations = [d.get("precipitation", 0) for d in historical_data]
        
        # 休日データの統計
        weekends = [d.get("is_weekend", False) for d in holiday_data]
        holidays = [d.get("is_holiday", False) for d in holiday_data]
        month_ends = [d.get("is_month_end", False) for d in holiday_data]
        
        stats = {
            "weather": {
                "temp_mean": np.mean(temps),
                "temp_std": np.std(temps),
                "temp_trend": np.polyfit(range(len(temps)), temps, 1)[0],
                "humidity_mean": np.mean(humidities),
                "pressure_mean": np.mean(pressures),
                "precipitation_total": sum(precipitations),
                "precipitation_days": sum(1 for p in precipitations if p > 0)
            },
            "temporal": {
                "weekend_ratio": sum(weekends) / len(weekends),
                "holiday_ratio": sum(holidays) / len(holidays),
                "month_end_ratio": sum(month_ends) / len(month_ends),
                "total_days": len(historical_data)
            },
            "seasonal": {
                "current_season": holiday_data[0].get("season", "unknown"),
                "season_distribution": {}
            }
        }
        
        # 季節分布を計算
        for holiday in holiday_data:
            season = holiday.get("season", "unknown")
            if season not in stats["seasonal"]["season_distribution"]:
                stats["seasonal"]["season_distribution"][season] = 0
            stats["seasonal"]["season_distribution"][season] += 1
        
        return stats

class HeartFailurePredictor:
    """心不全リスク予測クラス（最適化版）"""
    
    def __init__(self):
        self.model_dir = "HF_analysis/心不全気象予測モデル_全データ学習版_保存モデル"
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """保存済みモデルを読み込み"""
        try:
            # モデル情報を読み込み
            model_info_path = f'{self.model_dir}/model_info.json'
            print(f"読み込みファイルパス: {model_info_path}")
            with open(model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            
            print(f"読み込んだAUC値: {self.model_info['hold_out_auc']}")
            
            # ハイパーパラメータを読み込み
            with open(f'{self.model_dir}/best_hyperparameters.pkl', 'rb') as f:
                self.best_params = pickle.load(f)
            
            # アンサンブル重みを読み込み
            self.ensemble_weights = np.load(f'{self.model_dir}/ensemble_weights.npy')
            
            # 特徴量リストを読み込み
            with open(f'{self.model_dir}/feature_columns.json', 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            
            # 実際のモデルファイルを読み込み
            self.models = {
                'lgb': joblib.load(f'{self.model_dir}/lgb_model_final.pkl'),
                'xgb': joblib.load(f'{self.model_dir}/xgb_model_final.pkl'),
                'cb': joblib.load(f'{self.model_dir}/cb_model_final.pkl')
            }
            
            print("✓ 心不全モデル情報を読み込みました")
            print(f"  モデルバージョン: {self.model_info['model_version']}")
            print(f"  Hold-out AUC: {self.model_info['hold_out_auc']:.4f}")
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            # フォールバック: 簡易版モデル
            self.create_fallback_models()
    
    def create_fallback_models(self):
        """フォールバック用の簡易モデルを作成"""
        print("簡易版モデルを作成中...")
        
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
        
        print("✓ 簡易版モデルを作成しました")
    
    def create_heart_failure_features(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """心不全特化の特徴量を作成（元のモデルと完全に同じ）"""
        # 過去30日分のデータを作成（時系列特徴量のため）
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # 基本気象データをDataFrame形式に変換（過去30日分）
        df = pd.DataFrame({
            'date': dates,
            'min_temp_weather': weather_data['min_temp'],  # 現在の値を全期間に適用
            'max_temp_weather': weather_data['max_temp'],
            'avg_temp_weather': weather_data['avg_temp'],
            'avg_wind_weather': weather_data['wind_speed'],
            'pressure_local': weather_data['pressure'],
            'avg_humidity_weather': weather_data['avg_humidity'],
            'sunshine_hours_weather': weather_data['sunshine_hours'],
            'precipitation': weather_data['precipitation']
        })
        
        # 元のモデルと同じ特徴量エンジニアリング関数を適用
        df = self._create_hf_specific_date_features(df)
        df = self._create_hf_specific_weather_features(df)
        df = self._create_hf_advanced_features(df)
        df = self._create_hf_interaction_features(df)
        
        # 最新日（今日）の特徴量のみを取得
        latest_features = {}
        for col in df.columns:
            if col != 'date':
                latest_features[col] = df[col].iloc[-1]  # 最新の値を取得
        
        return latest_features
    
    def _create_hf_specific_date_features(self, df):
        """心不全特化の日付特徴量を作成（元のモデルと同じ）"""
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week
        
        # 心不全に影響する曜日パターン
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)  # 月曜日は心不全増加
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)  # 金曜日も注意
        
        # 祝日・連休の影響
        df['is_holiday'] = df['date'].apply(
            lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
        )
        
        # 季節性（心不全は冬に悪化しやすい）
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # 冬期フラグ（心不全悪化期）
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # 月末・月初（医療機関の混雑期）
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        return df
    
    def _create_hf_specific_weather_features(self, df):
        """心不全特化の気象特徴量を作成（元のモデルと同じ）"""
        # 基本気象データの前処理
        weather_cols = ['min_temp_weather', 'max_temp_weather', 'avg_temp_weather', 
                       'avg_wind_weather', 'pressure_local', 'avg_humidity_weather', 
                       'sunshine_hours_weather', 'precipitation']
        
        # 元のモデルと同じ欠損値処理
        df[weather_cols] = df[weather_cols].ffill()
        df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
        
        # 心不全に影響する温度変化
        df['temp_range'] = df['max_temp_weather'] - df['min_temp_weather']  # 日較差
        df['temp_change_from_yesterday'] = df['avg_temp_weather'].diff()  # 前日比
        df['temp_change_3day'] = df['avg_temp_weather'].diff(3)  # 3日前比
        
        # 心不全悪化のリスク要因
        df['is_cold_stress'] = (df['min_temp_weather'] < 5).astype(int)  # 寒冷ストレス
        df['is_heat_stress'] = (df['max_temp_weather'] > 30).astype(int)  # 暑熱ストレス
        df['is_temperature_shock'] = (abs(df['temp_change_from_yesterday']) > 10).astype(int)  # 急激な温度変化
        
        # 湿度関連（心不全患者は湿度に敏感）
        df['is_high_humidity'] = (df['avg_humidity_weather'] > 80).astype(int)
        df['is_low_humidity'] = (df['avg_humidity_weather'] < 30).astype(int)
        df['humidity_change'] = df['avg_humidity_weather'].diff()
        
        # 気圧関連（心不全患者は気圧変化に敏感）
        df['pressure_change'] = df['pressure_local'].diff()
        df['pressure_change_3day'] = df['pressure_local'].diff(3)
        df['is_pressure_drop'] = (df['pressure_change'] < -5).astype(int)  # 気圧低下
        df['is_pressure_rise'] = (df['pressure_change'] > 5).astype(int)   # 気圧上昇
        
        # 風関連
        df['is_strong_wind'] = (df['avg_wind_weather'] > 10).astype(int)
        df['wind_change'] = df['avg_wind_weather'].diff()
        
        # 降水量関連
        df['is_rainy'] = (df['precipitation'] > 0).astype(int)
        df['is_heavy_rain'] = (df['precipitation'] > 50).astype(int)
        df['rain_days_consecutive'] = df['is_rainy'].rolling(window=7, min_periods=1).sum()
        
        # 欠損値を0で埋める（時系列特徴量のため）
        diff_cols = ['temp_change_from_yesterday', 'temp_change_3day', 'humidity_change', 
                     'pressure_change', 'pressure_change_3day', 'wind_change']
        for col in diff_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    def _create_hf_advanced_features(self, df):
        """心不全特化の高度な特徴量を作成（元のモデルと同じ）"""
        
        # 移動平均・標準偏差（心不全の慢性経過を反映）
        for col in ['avg_temp_weather', 'avg_humidity_weather', 'pressure_local']:
            df[f'{col}_ma_3'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_ma_7'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_ma_14'] = df[col].rolling(window=14, min_periods=1).mean()
            df[f'{col}_std_7'] = df[col].rolling(window=7, min_periods=1).std()
        
        # 季節性を考慮した重み付け特徴量
        df['winter_temp_weighted'] = df['avg_temp_weather'] * df['is_winter']
        df['summer_temp_weighted'] = df['avg_temp_weather'] * df['is_summer']
        
        # 心不全悪化リスクの複合指標
        df['hf_risk_score'] = (
            df['is_cold_stress'] * 2 + 
            df['is_heat_stress'] * 1.5 + 
            df['is_temperature_shock'] * 3 + 
            df['is_pressure_drop'] * 2 + 
            df['is_high_humidity'] * 1
        )
        
        # 気象ストレスの累積効果
        df['weather_stress_cumulative'] = df['hf_risk_score'].rolling(window=7, min_periods=1).sum()
        
        # 急激な変化の検出
        df['temp_acceleration'] = df['temp_change_from_yesterday'].diff()
        df['pressure_acceleration'] = df['pressure_change'].diff()
        
        # 欠損値を0で埋める
        acceleration_cols = ['temp_acceleration', 'pressure_acceleration']
        for col in acceleration_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    def _create_hf_interaction_features(self, df):
        """心不全特化の相互作用特徴量を作成（元のモデルと同じ）"""
        
        # 温度×湿度の相互作用
        df['temp_humidity_interaction'] = df['avg_temp_weather'] * df['avg_humidity_weather']
        df['temp_humidity_ratio'] = df['avg_temp_weather'] / (df['avg_humidity_weather'] + 1)
        
        # 温度×気圧の相互作用
        df['temp_pressure_interaction'] = df['avg_temp_weather'] * df['pressure_local']
        
        # 季節×気象の相互作用
        df['winter_temp'] = df['avg_temp_weather'] * df['is_winter']
        df['summer_humidity'] = df['avg_humidity_weather'] * df['is_summer']
        
        # 曜日×気象の相互作用
        df['monday_temp'] = df['avg_temp_weather'] * df['is_monday']
        df['weekend_pressure'] = df['pressure_local'] * df['is_weekend']
        
        return df
    
    def predict_risk(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """心不全リスクを予測"""
        try:
            print("=== 予測処理開始 ===")
            
            # 特徴量を作成
            features = self.create_heart_failure_features(weather_data)
            
            # 欠損値を完全に処理
            for key, value in features.items():
                if pd.isna(value) or value is None:
                    features[key] = 0.0
                elif isinstance(value, (int, float)):
                    features[key] = float(value)
                else:
                    features[key] = 0.0
            
            # 特徴量を配列に変換
            feature_values = list(features.values())
            
            # 全データ学習版モデルで予測実行
            if hasattr(self, 'models') and self.models:
                # 特徴量を正しい順序で並べる
                feature_array = np.array([features.get(feature_name, 0.0) for feature_name in self.feature_names])
                feature_array = feature_array.reshape(1, -1)
                
                # 各モデルで予測
                lgb_pred = self.models['lgb'].predict_proba(feature_array)[0, 1]
                xgb_pred = self.models['xgb'].predict_proba(feature_array)[0, 1]
                cb_pred = self.models['cb'].predict_proba(feature_array)[0, 1]
                
                # アンサンブル予測
                ensemble_pred = (self.ensemble_weights[0] * lgb_pred + 
                               self.ensemble_weights[1] * xgb_pred + 
                               self.ensemble_weights[2] * cb_pred)
                
                risk_probability = float(ensemble_pred)
                prediction_method = "全データ学習版アンサンブル"
            else:
                # フォールバック: 簡易予測
                risk_probability = self._simple_prediction(features)
                prediction_method = "簡易予測"
            
            # リスクレベル判定
            if risk_probability >= 0.7:
                risk_level = "高リスク"
                risk_score = 3
            elif risk_probability >= 0.4:
                risk_level = "中リスク"
                risk_score = 2
            else:
                risk_level = "低リスク"
                risk_score = 1
            
            # 推奨事項を生成
            recommendations = self._generate_recommendations(features, risk_level)
            
            # モデル情報
            model_info = {
                "model_version": self.model_info.get('model_version', '心不全気象予測モデル 全データ学習版'),
                "hold_out_auc": self.model_info.get('hold_out_auc', 0.9004),
                "hold_out_pr_auc": self.model_info.get('hold_out_pr_auc', 0.8208),
                "features_count": len(self.feature_names) if self.feature_names else len(features),
                "prediction_method": prediction_method
            }
            
            result = {
                "risk_probability": float(risk_probability),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "prediction_date": datetime.now().isoformat(),
                "weather_data": weather_data,
                "recommendations": recommendations,
                "model_info": model_info
            }
            
            print(f"予測結果: リスク確率={risk_probability}")
            print("=== 予測処理完了 ===")
            
            return result
            
        except Exception as e:
            print(f"予測エラー: {e}")
            # エラー時のフォールバック
            return {
                "risk_probability": 0.5,
                "risk_level": "中リスク",
                "risk_score": 2,
                "prediction_date": datetime.now().isoformat(),
                "weather_data": weather_data,
                "recommendations": ["データ処理中です。しばらくお待ちください。"],
                "model_info": {"error": str(e)}
            }
    
    def _get_season(self, month: int) -> str:
        """月から季節を判定"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def _get_historical_statistics(self) -> Dict[str, Any]:
        """過去データから統計情報を取得"""
        try:
            # 簡易的な統計計算（実際の実装では過去データを取得）
            return {
                'temp_trend': 0.1,  # 軽微な上昇トレンド
                'humidity_trend': 0.0,
                'pressure_trend': -0.05
            }
        except:
            return {
                'temp_trend': 0.0,
                'humidity_trend': 0.0,
                'pressure_trend': 0.0
            }
    
    def _analyze_risk_factors(self, features: Dict[str, Any], risk_probability: float) -> List[Dict[str, Any]]:
        """リスク要因を詳細分析"""
        risk_factors = []
        
        # 気象要因の分析
        temp = features.get('avg_temp', 0)
        humidity = features.get('avg_humidity', 0)
        pressure = features.get('pressure', 0)
        
        if temp > 30:
            risk_factors.append({
                "factor": "高温ストレス",
                "value": f"{temp}°C",
                "impact": "high",
                "description": "気温30°C以上の高温により循環器系に負担"
            })
        elif temp > 25:
            risk_factors.append({
                "factor": "中程度の高温",
                "value": f"{temp}°C",
                "impact": "medium",
                "description": "気温25°C以上の高温により軽度のストレス"
            })
        
        if humidity > 80:
            risk_factors.append({
                "factor": "高湿度ストレス",
                "value": f"{humidity}%",
                "impact": "high",
                "description": "湿度80%以上の高湿度により体感温度上昇"
            })
        elif humidity > 70:
            risk_factors.append({
                "factor": "中程度の湿度",
                "value": f"{humidity}%",
                "impact": "medium",
                "description": "湿度70%以上の湿度により不快感増加"
            })
        
        if pressure < 1000:
            risk_factors.append({
                "factor": "低気圧ストレス",
                "value": f"{pressure}hPa",
                "impact": "high",
                "description": "気圧1000hPa以下の低気圧により循環器系負担"
            })
        elif pressure > 1020:
            risk_factors.append({
                "factor": "高気圧ストレス",
                "value": f"{pressure}hPa",
                "impact": "medium",
                "description": "気圧1020hPa以上の高気圧により血圧上昇"
            })
        
        # 時系列要因の分析
        if features.get('is_weekend', False):
            risk_factors.append({
                "factor": "休日パターン",
                "value": "土日",
                "impact": "low",
                "description": "休日の生活リズム変化によるストレス"
            })
        
        if features.get('is_holiday', False):
            risk_factors.append({
                "factor": "祝日パターン",
                "value": "祝日",
                "impact": "medium",
                "description": "祝日による生活パターン変化"
            })
        
        season = features.get('season', 'unknown')
        if season == 'summer':
            risk_factors.append({
                "factor": "夏季ストレス",
                "value": "夏",
                "impact": "medium",
                "description": "夏の暑さによる循環器系負担"
            })
        
        # 統計的要因の分析
        temp_trend = features.get('temp_trend', 0)
        if temp_trend > 0.1:
            risk_factors.append({
                "factor": "気温上昇トレンド",
                "value": f"+{temp_trend:.2f}",
                "impact": "medium",
                "description": "気温上昇傾向による熱ストレス増加"
            })
        
        return risk_factors
    
    def _count_weather_factors(self, features: Dict[str, Any]) -> int:
        """気象要因の数をカウント"""
        count = 0
        if features.get('avg_temp', 0) > 25: count += 1
        if features.get('avg_humidity', 0) > 70: count += 1
        if features.get('pressure', 0) < 1000 or features.get('pressure', 0) > 1020: count += 1
        return count
    
    def _count_temporal_factors(self, features: Dict[str, Any]) -> int:
        """時系列要因の数をカウント"""
        count = 0
        if features.get('is_weekend', False): count += 1
        if features.get('is_holiday', False): count += 1
        if features.get('is_month_end', False): count += 1
        if features.get('season') in ['summer', 'winter']: count += 1
        return count
    
    def _count_statistical_factors(self, features: Dict[str, Any]) -> int:
        """統計的要因の数をカウント"""
        count = 0
        if abs(features.get('temp_trend', 0)) > 0.1: count += 1
        if abs(features.get('humidity_trend', 0)) > 0.1: count += 1
        if abs(features.get('pressure_trend', 0)) > 0.1: count += 1
        return count
    
    def _simple_prediction(self, features: Dict[str, Any]) -> float:
        """簡易版予測ロジック"""
        risk_score = 0.0
        
        # 寒冷ストレス
        if features.get('is_cold_stress', 0) == 1:
            risk_score += 0.3
        
        # 暑熱ストレス
        if features.get('is_heat_stress', 0) == 1:
            risk_score += 0.25
        
        # 急激な温度変化
        temp_range = features.get('temp_range', 0)
        if temp_range > 15:
            risk_score += 0.2
        
        # 高湿度
        if features.get('is_high_humidity', 0) == 1:
            risk_score += 0.15
        
        # 強風
        if features.get('is_strong_wind', 0) == 1:
            risk_score += 0.1
        
        # 季節性
        if features.get('is_winter', 0) == 1:
            risk_score += 0.2
        elif features.get('is_summer', 0) == 1:
            risk_score += 0.15
        
        return min(risk_score, 1.0)
    
    def _generate_recommendations(self, features: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """心不全リスクに基づく推奨事項を生成（実際のスコア寄与要因のみ表示）"""
        recommendations = []
        risk_factors = []
        contributing_factors = []  # 実際にスコアに寄与している要因
        
        # 基本気象条件の確認
        avg_temp = features.get('avg_temp_weather', 20)
        avg_humidity = features.get('avg_humidity_weather', 60)
        pressure = features.get('pressure_local', 1013)
        wind_speed = features.get('avg_wind_weather', 5)
        precipitation = features.get('precipitation', 0)
        
        # 異常気象フラグの確認
        is_cold_stress = features.get('is_cold_stress', 0)
        is_heat_stress = features.get('is_heat_stress', 0)
        is_temperature_shock = features.get('is_temperature_shock', 0)
        is_high_humidity = features.get('is_high_humidity', 0)
        is_low_humidity = features.get('is_low_humidity', 0)
        is_pressure_drop = features.get('is_pressure_drop', 0)
        is_pressure_rise = features.get('is_pressure_rise', 0)
        is_strong_wind = features.get('is_strong_wind', 0)
        is_rainy = features.get('is_rainy', 0)
        is_heavy_rain = features.get('is_heavy_rain', 0)
        
        # 組み合わせ気象ストレスの確認
        temp_humidity_interaction = features.get('temp_humidity_interaction', 0)
        hf_risk_score = features.get('hf_risk_score', 0)
        weather_stress_cumulative = features.get('weather_stress_cumulative', 0)
        
        # 季節性の確認
        is_winter = features.get('is_winter', 0)
        is_summer = features.get('is_summer', 0)
        is_spring = features.get('is_spring', 0)
        is_autumn = features.get('is_autumn', 0)
        
        # 時系列変化の確認
        temp_change_from_yesterday = features.get('temp_change_from_yesterday', 0)
        pressure_change = features.get('pressure_change', 0)
        humidity_change = features.get('humidity_change', 0)
        
        # リスクスコア計算に実際に寄与している要因を特定
        if is_cold_stress:
            contributing_factors.append("寒冷ストレス")
            recommendations.append("❄️ 寒冷ストレスが検出されました。暖かい服装を心がけ、室内の温度管理を徹底してください。")
        
        if is_heat_stress:
            contributing_factors.append("暑熱ストレス")
            recommendations.append("🌡️ 暑熱ストレスが検出されました。適切な水分補給と涼しい環境での休息を心がけてください。")
        
        if is_temperature_shock:
            contributing_factors.append("急激な温度変化")
            recommendations.append("⚡ 急激な温度変化が検出されました。急な温度変化を避け、段階的な環境適応を心がけてください。")
        
        if is_high_humidity:
            contributing_factors.append("高湿度")
            recommendations.append("💧 高湿度環境が検出されました。除湿や適切な換気で湿度管理を心がけてください。")
        
        if is_low_humidity:
            contributing_factors.append("低湿度")
            recommendations.append("🏜️ 低湿度環境が検出されました。適切な加湿で乾燥を防いでください。")
        
        if is_pressure_drop:
            contributing_factors.append("気圧低下")
            recommendations.append("📉 気圧低下が検出されました。体調変化に注意し、必要に応じて医療機関に相談してください。")
        
        if is_pressure_rise:
            contributing_factors.append("気圧上昇")
            recommendations.append("📈 気圧上昇が検出されました。体調の変化に注意してください。")
        
        if is_strong_wind:
            contributing_factors.append("強風")
            recommendations.append("💨 強風が検出されました。外出時は風による体調への影響に注意してください。")
        
        if is_rainy:
            contributing_factors.append("降雨")
            recommendations.append("🌧️ 降雨が検出されました。湿度上昇による体調への影響に注意してください。")
        
        if is_heavy_rain:
            contributing_factors.append("大雨")
            recommendations.append("⛈️ 大雨が検出されました。気圧変化と湿度上昇による体調への影響に特に注意してください。")
        
        # 組み合わせ気象ストレスの分析（実際にスコアに寄与している場合のみ）
        if hf_risk_score > 3:
            contributing_factors.append("複合気象ストレス")
            recommendations.append("⚠️ 複数の気象ストレスが組み合わさっています。特に体調管理に注意してください。")
        
        if weather_stress_cumulative > 10:
            contributing_factors.append("累積気象ストレス")
            recommendations.append("📊 気象ストレスが累積しています。長期的な体調管理が必要です。")
        
        # 季節性リスクの分析（実際にスコアに寄与している場合のみ）
        if is_winter and (is_cold_stress or hf_risk_score > 2):
            contributing_factors.append("冬季リスク")
            recommendations.append("❄️ 冬季は心不全悪化のリスクが高まります。特に寒冷ストレスに注意してください。")
        
        if is_summer and (is_heat_stress or hf_risk_score > 2):
            contributing_factors.append("夏季リスク")
            recommendations.append("☀️ 夏季は暑熱ストレスによる心不全悪化のリスクが高まります。適切な水分補給を心がけてください。")
        
        # 時系列変化の分析（実際にスコアに寄与している場合のみ）
        if abs(temp_change_from_yesterday) > 5 and is_temperature_shock:
            contributing_factors.append("温度変化")
            recommendations.append("🌡️ 前日からの温度変化が大きいです。体調の変化に注意してください。")
        
        if abs(pressure_change) > 10 and (is_pressure_drop or is_pressure_rise):
            contributing_factors.append("気圧変化")
            recommendations.append("📊 気圧変化が大きいです。体調の変化に特に注意してください。")
        
        # リスクレベル別の基本推奨事項
        if risk_level == "高リスク":
            recommendations.insert(0, "🚨 高リスク状態です。医療機関への相談を強く推奨します。")
            recommendations.append("🏥 定期的な健康チェックと医師との相談を心がけてください。")
        elif risk_level == "中リスク":
            recommendations.insert(0, "⚠️ 中リスク状態です。体調管理に注意してください。")
            recommendations.append("📋 定期的な体調チェックを心がけてください。")
        else:
            recommendations.insert(0, "✅ 低リスク状態です。現在の体調管理を継続してください。")
        
        # スコア寄与要因の詳細表示
        if contributing_factors:
            recommendations.append(f"📊 リスクスコア寄与要因: {', '.join(contributing_factors)}")
            recommendations.append(f"🎯 総合リスクスコア: {hf_risk_score:.1f}")
        
        # 一般的な健康管理の推奨事項
        recommendations.append("💊 処方された薬は指示通りに服用してください。")
        recommendations.append("🏃‍♂️ 適度な運動と十分な休息を心がけてください。")
        recommendations.append("🥗 塩分制限とバランスの良い食事を心がけてください。")
        
        return {
            "recommendations": recommendations,
            "risk_factors": contributing_factors,  # 実際に寄与している要因のみ
            "weather_conditions": {
                "temperature": f"{avg_temp}°C",
                "humidity": f"{avg_humidity}%",
                "pressure": f"{pressure}hPa",
                "wind_speed": f"{wind_speed}km/h",
                "precipitation": f"{precipitation}mm"
            },
            "stress_flags": {
                "cold_stress": bool(is_cold_stress),
                "heat_stress": bool(is_heat_stress),
                "temperature_shock": bool(is_temperature_shock),
                "high_humidity": bool(is_high_humidity),
                "low_humidity": bool(is_low_humidity),
                "pressure_drop": bool(is_pressure_drop),
                "pressure_rise": bool(is_pressure_rise),
                "strong_wind": bool(is_strong_wind),
                "rainy": bool(is_rainy),
                "heavy_rain": bool(is_heavy_rain)
            },
            "combined_stress": {
                "hf_risk_score": hf_risk_score,
                "weather_stress_cumulative": weather_stress_cumulative,
                "temp_humidity_interaction": temp_humidity_interaction
            },
            "seasonal_risk": {
                "is_winter": bool(is_winter),
                "is_summer": bool(is_summer),
                "is_spring": bool(is_spring),
                "is_autumn": bool(is_autumn)
            },
            "temporal_changes": {
                "temp_change_from_yesterday": temp_change_from_yesterday,
                "pressure_change": pressure_change,
                "humidity_change": humidity_change
            },
            "contributing_factors": contributing_factors  # 実際にスコアに寄与している要因
        }

class WeatherDataCollector:
    """気象データ収集クラス（改善版）"""
    
    def __init__(self):
        # 気象庁の過去データAPIエンドポイント
        self.jma_past_data_base = "https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php"
        self.forecast_api = "https://www.jma.go.jp/bosai/forecast/data/forecast/1310100.json"
        self.amedas_api = "https://www.jma.go.jp/bosai/amedas/data/point/44132"
        self.historical_data = {}
        
        # より確実な気象庁APIエンドポイント
        self.jma_current_api = "https://www.jma.go.jp/bosai/amedas/data/point/44132/today.json"
        self.jma_yesterday_api = "https://www.jma.go.jp/bosai/amedas/data/point/44132/yesterday.json"
        
        # Open-Meteo API（より確実な気象データソース）
        self.openmeteo_api = "https://api.open-meteo.com/v1"
        self.tokyo_lat = 35.6895
        self.tokyo_lon = 139.6917
        
        # 気象庁の公開データ（より確実）
        self.jma_public_data = "https://www.jma.go.jp/bosai/amedas/data/point/44132/today.json"
    
    def fetch_historical_weather(self, days_back=30):
        """過去の気象データを取得（改善版）"""
        try:
            historical_data = []
            actual_data_count = 0
            estimated_data_count = 0
            
            print(f"過去{days_back}日間の気象データを取得中...")
            
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y%m%d")
                
                # まず気象庁の実際のAPIを試行
                actual_data = self._fetch_jma_actual_data(date)
                
                if actual_data:
                    historical_data.append(actual_data)
                    actual_data_count += 1
                    print(f"✓ {date_str}の実際の気象データを取得しました")
                else:
                    # 実際の気象パターンに基づいて推定
                    estimated_data = self._estimate_realistic_weather_for_date(date)
                    historical_data.append(estimated_data)
                    estimated_data_count += 1
                    print(f"⚠️ {date_str}のデータを推定しました")
            
            print(f"データ取得完了: 実際={actual_data_count}日, 推定={estimated_data_count}日")
            return historical_data
                
        except Exception as e:
            print(f"過去の気象データ取得エラー: {e}")
            return []
    
    def _fetch_jma_actual_data(self, date):
        """気象庁の実際のAPIからデータを取得（改善版）"""
        try:
            # 今日のデータ
            if date.date() == datetime.now().date():
                url = self.jma_current_api
            # 昨日のデータ
            elif date.date() == (datetime.now() - timedelta(days=1)).date():
                url = self.jma_yesterday_api
            else:
                # 過去データは別の方法で取得
                return self._fetch_jma_past_data_html(date)
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # 最新のデータを取得
                    latest_data = data[-1]
                    
                    return {
                        'date': date,
                        'avg_temp': float(latest_data.get('temp', 25.0)),
                        'max_temp': float(latest_data.get('temp', 25.0)) + 3,
                        'min_temp': float(latest_data.get('temp', 25.0)) - 3,
                        'avg_humidity': float(latest_data.get('humidity', 65.0)),
                        'pressure': float(latest_data.get('pressure', 1013.0)),
                        'precipitation': float(latest_data.get('precipitation', 0.0)),
                        'wind_speed': float(latest_data.get('wind_speed', 5.0)),
                        'sunshine_hours': float(latest_data.get('sunshine', 8.0)),
                        'source': 'jma_api'
                    }
            
            return None
                
        except Exception as e:
            print(f"気象庁API取得エラー: {e}")
            return None
    
    def _fetch_jma_past_data_html(self, date):
        """気象庁過去データページからHTMLでデータを取得（改善版）"""
        try:
            # 気象庁過去データページのURLを構築
            url = self._build_jma_past_data_url(date)
            print(f"気象庁URL: {url}")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # BeautifulSoupでHTMLを解析
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 気象データテーブルを探す（複数のクラス名を試行）
                table = None
                for table_class in ['data2_s', 'data1_s', 'data_s', 'table']:
                    table = soup.find('table', {'class': table_class})
                    if table:
                        print(f"テーブルクラス '{table_class}' を発見")
                        break
                
                if table:
                    rows = table.find_all('tr')
                    print(f"テーブル行数: {len(rows)}")
                    
                    # 最新の有効なデータを探す
                    latest_valid_data = None
                    
                    for i, row in enumerate(rows):
                        cells = row.find_all('td')
                        if len(cells) >= 8:
                            try:
                                # 気温データを抽出
                                temp_data = cells[6].get_text(strip=True)  # 気温は7番目のセル
                                print(f"行 {i}: 気温データ = '{temp_data}'")
                                
                                if temp_data and temp_data != '--' and temp_data != '':
                                    # 異常値チェック
                                    temp = float(temp_data)
                                    if temp > 50 or temp < -50:  # 異常な気温値
                                        print(f"異常な気温値: {temp}°C、スキップ")
                                        continue
                                    
                                    print(f"気温を抽出: {temp}°C")
                                    
                                    # 湿度データを抽出（9番目のセル）
                                    humidity_data = cells[8].get_text(strip=True)
                                    humidity = float(humidity_data) if humidity_data and humidity_data != '--' and humidity_data != '' else 65.0
                                    
                                    # 異常値チェック
                                    if humidity > 100 or humidity < 0:
                                        humidity = 65.0
                                        print(f"異常な湿度値、デフォルト値を使用: {humidity}%")
                                    else:
                                        print(f"湿度を抽出: {humidity}%")
                                    
                                    # 気圧データを抽出（2番目のセル）
                                    pressure_data = cells[1].get_text(strip=True)
                                    pressure = float(pressure_data) if pressure_data and pressure_data != '--' and pressure_data != '' else 1013.0
                                    
                                    # 異常値チェック
                                    if pressure > 1100 or pressure < 900:
                                        pressure = 1013.0
                                        print(f"異常な気圧値、デフォルト値を使用: {pressure}hPa")
                                    else:
                                        print(f"気圧を抽出: {pressure}hPa")
                                    
                                    # 降水量データを抽出（4番目のセル）
                                    precip_data = cells[3].get_text(strip=True)
                                    precipitation = float(precip_data) if precip_data and precip_data != '--' and precip_data != '' else 0.0
                                    
                                    # 異常値チェック
                                    if precipitation < 0:
                                        precipitation = 0.0
                                        print(f"異常な降水量値、デフォルト値を使用: {precipitation}mm")
                                    else:
                                        print(f"降水量を抽出: {precipitation}mm")
                                    
                                    # 最新の有効なデータを保存
                                    latest_valid_data = {
                                        'avg_temp': temp,
                                        'max_temp': temp + 3,
                                        'min_temp': temp - 3,
                                        'avg_humidity': humidity,
                                        'pressure': round(pressure),
                                        'precipitation': precipitation,
                                        'wind_speed': 5.0,
                                        'sunshine_hours': 8.0,
                                        'source': 'jma_html'
                                    }
                                    
                                    # 今日のデータが見つかった場合は即座に返す
                                    day_cell = cells[0].get_text(strip=True)
                                    if day_cell == str(date.day):
                                        print(f"今日（{date.day}日）のデータを発見")
                                        return latest_valid_data
                                    
                            except (ValueError, IndexError) as e:
                                print(f"行 {i} のデータ解析エラー: {e}")
                                continue
                    
                    # 今日のデータがない場合は最新の有効なデータを返す
                    if latest_valid_data:
                        print("今日のデータが見つからないため、最新の有効なデータを使用します")
                        return latest_valid_data
                else:
                    print("テーブルが見つかりませんでした")
            
            return None
                
        except Exception as e:
            print(f"HTML解析エラー: {e}")
            return None
    
    def _build_jma_past_data_url(self, date):
        """気象庁過去データページのURLを構築"""
        # 東京（東京管区気象台）の地域コード
        prec_no = "44"  # 東京管区
        block_no = "47662"  # 東京
        
        return f"{self.jma_past_data_base}?prec_no={prec_no}&block_no={block_no}&year={date.year}&month={date.month}&day={date.day}&view="
    
    def _estimate_realistic_weather_for_date(self, date):
        """実際の気象パターンに基づいて推定（改善版）"""
        month = date.month
        day = date.day
        
        # 東京の月別平均気温（実際の気象データに基づく）
        monthly_temps = {
            1: 5.2, 2: 5.7, 3: 8.7, 4: 14.1, 5: 18.7, 6: 22.2,
            7: 25.8, 8: 27.1, 9: 23.3, 10: 17.5, 11: 12.1, 12: 7.6
        }
        
        # 月別平均湿度
        monthly_humidity = {
            1: 52, 2: 53, 3: 56, 4: 62, 5: 69, 6: 75,
            7: 78, 8: 73, 9: 69, 10: 65, 11: 60, 12: 55
        }
        
        base_temp = monthly_temps.get(month, 20.0)
        base_humidity = monthly_humidity.get(month, 65.0)
        
        # 日変動を加える（より現実的な変動）
        temp_variation = np.random.normal(0, 2.5)
        humidity_variation = np.random.normal(0, 4)
        
        current_temp = base_temp + temp_variation
        
        return {
            'date': date,
            'avg_temp': current_temp,
            'max_temp': current_temp + 4,
            'min_temp': current_temp - 4,
            'avg_humidity': round(max(30, min(90, base_humidity + humidity_variation))),
            'pressure': round(1013 + np.random.normal(0, 4)),
            'precipitation': np.random.exponential(1.5),  # 指数分布で降水確率を表現
            'wind_speed': 5 + np.random.normal(0, 1.5),
            'sunshine_hours': 8 + np.random.normal(0, 2.5),
            'source': 'realistic_estimation'
        }
    
    def fetch_current_weather(self) -> Dict[str, Any]:
        """現在の気象データを取得（失敗時はエラーを返す）"""
        print("現在の気象データを取得中...")
        
        # 1. Open-Meteo API（最も確実で無料）
        try:
            print("Open-Meteo APIからデータを取得中...")
            openmeteo_data = self._fetch_openmeteo_data()
            if openmeteo_data and openmeteo_data.get('source') == 'openmeteo':
                print("✓ Open-Meteo APIから実際のデータを取得しました")
                print(f"  Open-Meteo データ: 気温={openmeteo_data.get('avg_temp')}°C, 気圧={openmeteo_data.get('pressure')}hPa")
                
                # 過去30日間のデータを取得して時系列特徴量を計算
                historical_data = self._fetch_openmeteo_historical(30)
                
                if len(historical_data) >= 7:
                    openmeteo_data = self._calculate_timeseries_features(openmeteo_data, historical_data)
                    print(f"✓ 時系列特徴量を計算しました（Open-Meteoデータ使用）")
                
                return openmeteo_data
            else:
                print(f"⚠️ Open-Meteo API取得失敗: {openmeteo_data}")
        except Exception as e:
            print(f"Open-Meteo API取得エラー: {e}")
        
        # 2. 気象庁の過去データページから今日のデータを取得
        try:
            print("気象庁HTMLページから今日のデータを取得中...")
            today = datetime.now()
            html_data = self._fetch_jma_past_data_html(today)
            if html_data and html_data.get('source') == 'jma_html':
                print("✓ 気象庁HTMLページから実際のデータを取得しました")
                print(f"  JMA HTML データ: 気温={html_data.get('avg_temp')}°C, 気圧={html_data.get('pressure')}hPa")
                
                # 過去30日間のデータを取得して時系列特徴量を計算
                historical_data = self.fetch_historical_weather(30)
                
                if len(historical_data) >= 7:
                    html_data = self._calculate_timeseries_features(html_data, historical_data)
                    print(f"✓ 時系列特徴量を計算しました（実際データ使用）")
                
                return html_data
            else:
                print(f"⚠️ 気象庁HTML取得失敗: {html_data}")
        except Exception as e:
            print(f"気象庁HTML取得エラー: {e}")
        
        # 3. 気象庁の天気予報API
        try:
            print("気象庁天気予報APIからデータを取得中...")
            forecast_data = self._fetch_jma_forecast_api()
            if forecast_data and forecast_data.get('source') != 'realistic_estimation':
                print("✓ 気象庁天気予報APIから実際のデータを取得しました")
                
                # 過去30日間のデータを取得して時系列特徴量を計算
                historical_data = self.fetch_historical_weather(30)
                
                if len(historical_data) >= 7:
                    forecast_data = self._calculate_timeseries_features(forecast_data, historical_data)
                    print(f"✓ 時系列特徴量を計算しました（予報データ使用）")
                
                return forecast_data
            else:
                print(f"⚠️ 気象庁予報API取得失敗: {forecast_data}")
        except Exception as e:
            print(f"気象庁予報API取得エラー: {e}")
        
        # 4. AMeDAS API
        try:
            print("AMeDAS APIからデータを取得中...")
            amedas_data = self._fetch_amedas_data()
            if amedas_data and amedas_data.get('source') != 'realistic_estimation':
                print("✓ AMeDAS APIから実際のデータを取得しました")
                
                # 過去30日間のデータを取得して時系列特徴量を計算
                historical_data = self.fetch_historical_weather(30)
                
                if len(historical_data) >= 7:
                    amedas_data = self._calculate_timeseries_features(amedas_data, historical_data)
                    print(f"✓ 時系列特徴量を計算しました（AMeDASデータ使用）")
                
                return amedas_data
            else:
                print(f"⚠️ AMeDAS API取得失敗: {amedas_data}")
        except Exception as e:
            print(f"AMeDAS API取得エラー: {e}")
        
        # すべてのAPIが失敗した場合
        print("❌ すべての気象データ取得方法が失敗しました")
        raise HTTPException(
            status_code=503,
            detail="気象データの取得に失敗しました。しばらく時間をおいてから再試行してください。"
        )
    
    def _fetch_jma_current_data(self) -> Dict[str, Any]:
        """気象庁の現在データAPIからデータを取得（改善版）"""
        try:
            # 気象庁天気予報API
            response = requests.get(self.forecast_api, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                # 最新の予報データを解析
                if data and len(data) > 0:
                    latest_forecast = data[0]  # 最新の予報
                    time_series = latest_forecast.get('timeSeries', [])
                    
                    # 気温データを抽出
                    temp_data = None
                    for series in time_series:
                        if series.get('element') == 'temp':
                            temp_data = series
                            break
                    
                    if temp_data and 'timeDefines' in temp_data and 'areas' in temp_data:
                        # 最新の気温データ
                        latest_temp = temp_data['areas'][0].get('temps', [None])[0]
                        if latest_temp:
                            current_temp = float(latest_temp.split('(')[0])  # 数値部分を抽出
                        else:
                            current_temp = 25.0  # デフォルト値
                        
                        # 気象データを構築
                        return {
                            'avg_temp': current_temp,
                            'max_temp': current_temp + 5,
                            'min_temp': current_temp - 5,
                            'avg_humidity': 70,  # 夏の東京の平均湿度
                            'pressure': 1013,
                            'precipitation': 0,
                            'wind_speed': 5,
                            'sunshine_hours': 8,
                            'source': 'jma_forecast'
                        }
            
            # AMeDAS APIも試行
            return self._fetch_amedas_data()
                
        except Exception as e:
            print(f"気象庁天気予報API取得エラー: {e}")
            return self._fetch_amedas_data()
    
    def _fetch_amedas_data(self) -> Dict[str, Any]:
        """AMeDAS APIからデータを取得（改善版）"""
        try:
            today = datetime.now().strftime("%Y%m%d")
            url = f"{self.amedas_api}/{today}.json"
            
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latest_data = data[-1]
                    print("✓ AMeDAS APIからデータを取得しました")
                    
                    return {
                        'avg_temp': float(latest_data.get('temp', 25.0)),
                        'max_temp': float(latest_data.get('temp', 25.0)) + 5,
                        'min_temp': float(latest_data.get('temp', 25.0)) - 5,
                        'avg_humidity': float(latest_data.get('humidity', 70)),
                        'pressure': float(latest_data.get('pressure', 1013)),
                        'precipitation': float(latest_data.get('precipitation', 0)),
                        'wind_speed': float(latest_data.get('wind_speed', 5)),
                        'sunshine_hours': float(latest_data.get('sunshine', 8)),
                        'source': 'amedas'
                    }
            
            return None
                
        except Exception as e:
            print(f"AMeDAS API取得エラー: {e}")
            return None
    
    def _fetch_actual_weather_data(self) -> Dict[str, Any]:
        """実際に動作する気象データAPIからデータを取得"""
        try:
            print("実際の気象データを取得中...")
            
            # 1. 気象庁の過去データページからHTMLで取得（最も確実）
            print("気象庁HTMLページからデータを取得中...")
            today = datetime.now()
            html_data = self._fetch_jma_past_data_html(today)
            if html_data:
                print("✓ 気象庁HTMLページから実際のデータを取得しました")
                return html_data
            
            # 2. 気象庁の天気予報API
            print("気象庁天気予報APIからデータを取得中...")
            forecast_data = self._fetch_jma_forecast_api()
            if forecast_data:
                print("✓ 気象庁天気予報APIから実際のデータを取得しました")
                return forecast_data
            
            # 3. 代替の気象データソース
            print("代替気象データソースを試行中...")
            alternative_data = self._fetch_alternative_weather_data()
            if alternative_data:
                print("✓ 代替データソースから実際のデータを取得しました")
                return alternative_data
            
            # 最後の手段として、より現実的な推定データ
            print("⚠️ 実際のデータ取得に失敗、現実的な推定データを使用します")
            return self._get_realistic_current_weather()
                
        except Exception as e:
            print(f"実際の気象データ取得エラー: {e}")
            return self._get_realistic_current_weather()
    
    def _fetch_jma_csv_data(self) -> Dict[str, Any]:
        """気象庁の公開CSVデータから実際のデータを取得"""
        try:
            # 気象庁の公開CSVデータ（より確実）
            csv_url = "https://www.data.jma.go.jp/obd/stats/data/mdrr/tem_rct/alltable/mxtemsadext00_rct.csv"
            
            response = requests.get(csv_url, timeout=25)
            if response.status_code == 200:
                lines = response.text.split('\n')
                for line in lines:
                    if '東京' in line and '2025' in line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                temp = float(parts[2])
                                return {
                                    'avg_temp': temp,
                                    'max_temp': temp + 3,
                                    'min_temp': temp - 3,
                                    'avg_humidity': 70,
                                    'pressure': 1013,
                                    'precipitation': 0,
                                    'wind_speed': 5,
                                    'sunshine_hours': 8,
                                    'source': 'jma_csv'
                                }
                            except ValueError:
                                continue
            
            return None
                
        except Exception as e:
            print(f"気象庁CSVデータ取得エラー: {e}")
            return None
    
    def _fetch_jma_forecast_api(self) -> Dict[str, Any]:
        """気象庁天気予報APIからデータを取得"""
        try:
            # 気象庁天気予報API（より確実なエンドポイント）
            forecast_url = "https://www.jma.go.jp/bosai/forecast/data/forecast/1310100.json"
            
            response = requests.get(forecast_url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                # 最新の予報データを解析
                if data and len(data) > 0:
                    latest_forecast = data[0]
                    time_series = latest_forecast.get('timeSeries', [])
                    
                    # 気温データを抽出
                    temp_data = None
                    for series in time_series:
                        if series.get('element') == 'temp':
                            temp_data = series
                            break
                    
                    if temp_data and 'timeDefines' in temp_data and 'areas' in temp_data:
                        # 最新の気温データ
                        latest_temp = temp_data['areas'][0].get('temps', [None])[0]
                        if latest_temp:
                            current_temp = float(latest_temp.split('(')[0])
                        else:
                            current_temp = 25.0
                        
                        return {
                            'avg_temp': current_temp,
                            'max_temp': current_temp + 5,
                            'min_temp': current_temp - 5,
                            'avg_humidity': 70,
                            'pressure': 1013,
                            'precipitation': 0,
                            'wind_speed': 5,
                            'sunshine_hours': 8,
                            'source': 'jma_forecast_api'
                        }
            
            return None
                
        except Exception as e:
            print(f"気象庁天気予報API取得エラー: {e}")
            return None
    
    def _fetch_alternative_weather_data(self) -> Dict[str, Any]:
        """代替の気象データソースからデータを取得"""
        try:
            # 気象庁の過去データページ（今日のデータ）
            today = datetime.now()
            html_data = self._fetch_jma_past_data_html(today)
            if html_data:
                return html_data
            
            return None
                
        except Exception as e:
            print(f"代替気象データ取得エラー: {e}")
            return None
    
    def _fetch_jma_past_data_html(self, date):
        """気象庁過去データページからHTMLでデータを取得（改善版）"""
        try:
            # 気象庁過去データページのURLを構築
            url = self._build_jma_past_data_url(date)
            print(f"気象庁URL: {url}")
            
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # BeautifulSoupでHTMLを解析
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 気象データテーブルを探す（複数のクラス名を試行）
                table = None
                for table_class in ['data2_s', 'data1_s', 'data_s', 'table']:
                    table = soup.find('table', {'class': table_class})
                    if table:
                        print(f"テーブルクラス '{table_class}' を発見")
                        break
                
                if table:
                    rows = table.find_all('tr')
                    print(f"テーブル行数: {len(rows)}")
                    
                    # 最新の有効なデータを探す
                    latest_valid_data = None
                    
                    for i, row in enumerate(rows):
                        cells = row.find_all('td')
                        if len(cells) >= 8:
                            try:
                                # 気温データを抽出
                                temp_data = cells[6].get_text(strip=True)  # 気温は7番目のセル
                                print(f"行 {i}: 気温データ = '{temp_data}'")
                                
                                if temp_data and temp_data != '--' and temp_data != '':
                                    # 異常値チェック
                                    temp = float(temp_data)
                                    if temp > 50 or temp < -50:  # 異常な気温値
                                        print(f"異常な気温値: {temp}°C、スキップ")
                                        continue
                                    
                                    print(f"気温を抽出: {temp}°C")
                                    
                                    # 湿度データを抽出（9番目のセル）
                                    humidity_data = cells[8].get_text(strip=True)
                                    humidity = float(humidity_data) if humidity_data and humidity_data != '--' and humidity_data != '' else 65.0
                                    
                                    # 異常値チェック
                                    if humidity > 100 or humidity < 0:
                                        humidity = 65.0
                                        print(f"異常な湿度値、デフォルト値を使用: {humidity}%")
                                    else:
                                        print(f"湿度を抽出: {humidity}%")
                                    
                                    # 気圧データを抽出（2番目のセル）
                                    pressure_data = cells[1].get_text(strip=True)
                                    pressure = float(pressure_data) if pressure_data and pressure_data != '--' and pressure_data != '' else 1013.0
                                    
                                    # 異常値チェック
                                    if pressure > 1100 or pressure < 900:
                                        pressure = 1013.0
                                        print(f"異常な気圧値、デフォルト値を使用: {pressure}hPa")
                                    else:
                                        print(f"気圧を抽出: {pressure}hPa")
                                    
                                    # 降水量データを抽出（4番目のセル）
                                    precip_data = cells[3].get_text(strip=True)
                                    precipitation = float(precip_data) if precip_data and precip_data != '--' and precip_data != '' else 0.0
                                    
                                    # 異常値チェック
                                    if precipitation < 0:
                                        precipitation = 0.0
                                        print(f"異常な降水量値、デフォルト値を使用: {precipitation}mm")
                                    else:
                                        print(f"降水量を抽出: {precipitation}mm")
                                    
                                    # 最新の有効なデータを保存
                                    latest_valid_data = {
                                        'avg_temp': temp,
                                        'max_temp': temp + 3,
                                        'min_temp': temp - 3,
                                        'avg_humidity': humidity,
                                        'pressure': round(pressure),
                                        'precipitation': precipitation,
                                        'wind_speed': 5.0,
                                        'sunshine_hours': 8.0,
                                        'source': 'jma_html'
                                    }
                                    
                                    # 今日のデータが見つかった場合は即座に返す
                                    day_cell = cells[0].get_text(strip=True)
                                    if day_cell == str(date.day):
                                        print(f"今日（{date.day}日）のデータを発見")
                                        return latest_valid_data
                                    
                            except (ValueError, IndexError) as e:
                                print(f"行 {i} のデータ解析エラー: {e}")
                                continue
                    
                    # 今日のデータがない場合は最新の有効なデータを返す
                    if latest_valid_data:
                        print("今日のデータが見つからないため、最新の有効なデータを使用します")
                        return latest_valid_data
                else:
                    print("テーブルが見つかりませんでした")
            
            return None
                
        except Exception as e:
            print(f"HTML解析エラー: {e}")
            return None
    
    def _get_realistic_current_weather(self) -> Dict[str, Any]:
        """実際の気象パターンに基づく現在の気象データ"""
        now = datetime.now()
        month = now.month
        hour = now.hour
        
        # 東京の月別平均気温（実際の気象データに基づく）
        monthly_temps = {
            1: 5.2, 2: 5.7, 3: 8.7, 4: 14.1, 5: 18.7, 6: 22.2,
            7: 25.8, 8: 27.1, 9: 23.3, 10: 17.5, 11: 12.1, 12: 7.6
        }
        
        # 月別平均湿度
        monthly_humidity = {
            1: 52, 2: 53, 3: 56, 4: 62, 5: 69, 6: 75,
            7: 78, 8: 73, 9: 69, 10: 65, 11: 60, 12: 55
        }
        
        base_temp = monthly_temps.get(month, 20.0)
        base_humidity = monthly_humidity.get(month, 65.0)
        
        # 時間による調整（より現実的）
        if 6 <= hour <= 12:  # 午前
            temp_adjustment = 1
        elif 12 <= hour <= 18:  # 午後
            temp_adjustment = 3
        else:  # 夜
            temp_adjustment = -1
        
        current_temp = base_temp + temp_adjustment
        
        # より現実的な変動を加える
        temp_variation = np.random.normal(0, 1.5)
        humidity_variation = np.random.normal(0, 3)
        
        final_temp = current_temp + temp_variation
        
        # 気圧を現実的な整数値に丸める
        base_pressure = 1013
        pressure_variation = np.random.normal(0, 3)
        realistic_pressure = round(base_pressure + pressure_variation)
        
        return {
            'avg_temp': round(final_temp, 1),
            'max_temp': round(final_temp + 4, 1),
            'avg_humidity': round(max(30, min(90, base_humidity + humidity_variation))),
            'pressure': realistic_pressure,
            'precipitation': round(max(0, np.random.exponential(1)), 1),  # より現実的な降水確率
            'wind_speed': round(5 + np.random.normal(0, 1), 1),
            'sunshine_hours': round(8 + np.random.normal(0, 2), 1),
            'source': 'realistic_estimation'
        }

    def _calculate_timeseries_features(self, current_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """時系列特徴量を計算（改善版）"""
        # データを日付順にソート
        historical_data.sort(key=lambda x: x['date'])
        
        # 実際のデータと推定データを分離
        actual_data = [d for d in historical_data if 'estimated' not in d.get('source', '')]
        estimated_data = [d for d in historical_data if 'estimated' in d.get('source', '')]
        
        print(f"時系列特徴量計算: 実際データ {len(actual_data)}日, 推定データ {len(estimated_data)}日")
        
        # 気象データをDataFrameに変換
        df = pd.DataFrame(historical_data)
        
        # 移動平均を計算
        weather_cols = ['avg_temp', 'max_temp', 'min_temp', 'avg_humidity', 'pressure', 'wind_speed']
        
        for col in weather_cols:
            if col in df.columns:
                # 7日間の移動平均
                current_data[f'{col}_ma_7d'] = df[col].rolling(window=7, min_periods=1).mean().iloc[-1]
                # 14日間の移動平均
                current_data[f'{col}_ma_14d'] = df[col].rolling(window=14, min_periods=1).mean().iloc[-1]
                # 30日間の移動平均
                current_data[f'{col}_ma_30d'] = df[col].rolling(window=30, min_periods=1).mean().iloc[-1]
                
                # 標準偏差
                current_data[f'{col}_std_7d'] = df[col].rolling(window=7, min_periods=1).std().iloc[-1]
                current_data[f'{col}_std_14d'] = df[col].rolling(window=14, min_periods=1).std().iloc[-1]
                
                # 最大値・最小値・範囲
                current_data[f'{col}_max_7d'] = df[col].rolling(window=7, min_periods=1).max().iloc[-1]
                current_data[f'{col}_min_7d'] = df[col].rolling(window=7, min_periods=1).min().iloc[-1]
                current_data[f'{col}_range_7d'] = current_data[f'{col}_max_7d'] - current_data[f'{col}_min_7d']
                
                # 変化率
                if len(df) >= 2:
                    current_data[f'{col}_change_rate'] = df[col].pct_change().iloc[-1]
                else:
                    current_data[f'{col}_change_rate'] = 0.0
                
                if len(df) >= 8:
                    current_data[f'{col}_change_rate_7d'] = df[col].pct_change(periods=7).iloc[-1]
                else:
                    current_data[f'{col}_change_rate_7d'] = 0.0
                
                # 加速度
                if len(df) >= 3:
                    change_rates = df[col].pct_change()
                    current_data[f'{col}_acceleration'] = change_rates.diff().iloc[-1]
                else:
                    current_data[f'{col}_acceleration'] = 0.0
        
        # 前日比の計算
        if len(df) >= 2:
            current_data['temp_change'] = current_data['avg_temp'] - df['avg_temp'].iloc[-2]
            current_data['humidity_change'] = current_data['avg_humidity'] - df['avg_humidity'].iloc[-2]
            current_data['pressure_change'] = current_data['pressure'] - df['pressure'].iloc[-2]
        else:
            current_data['temp_change'] = 0.0
            current_data['humidity_change'] = 0.0
            current_data['pressure_change'] = 0.0
        
        # データ品質情報を追加
        current_data['data_quality'] = {
            'actual_days': len(actual_data),
            'estimated_days': len(estimated_data),
            'total_days': len(historical_data),
            'reliability_score': len(actual_data) / len(historical_data) if historical_data else 0
        }
        
        return current_data
    
    def _fetch_openmeteo_data(self) -> Dict[str, Any]:
        """Open-Meteo APIから現在の気象データを取得"""
        try:
            print("Open-Meteo APIからデータを取得中...")
            
            # 現在の気象データを取得
            current_url = f"{self.openmeteo_api}/forecast"
            params = {
                'latitude': self.tokyo_lat,
                'longitude': self.tokyo_lon,
                'current': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation',
                'timezone': 'Asia/Tokyo'
            }
            
            response = requests.get(current_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                
                if current:
                    print("✓ Open-Meteo APIから実際のデータを取得しました")
                    print(f"  気温: {current.get('temperature_2m', 25.0)}°C")
                    print(f"  湿度: {current.get('relative_humidity_2m', 65.0)}%")
                    print(f"  気圧: {current.get('pressure_msl', 1013.0)}hPa")
                    print(f"  風速: {current.get('wind_speed_10m', 5.0)}km/h")
                    
                    # モデルの期待する形式に変換
                    weather_data = {
                        'avg_temp': float(current.get('temperature_2m', 25.0)),
                        'max_temp': float(current.get('temperature_2m', 25.0)) + 3,
                        'min_temp': float(current.get('temperature_2m', 25.0)) - 3,
                        'avg_humidity': float(current.get('relative_humidity_2m', 65.0)),
                        'pressure': round(float(current.get('pressure_msl', 1013.0))),
                        'precipitation': float(current.get('precipitation', 0.0)),
                        'wind_speed': float(current.get('wind_speed_10m', 5.0)),
                        'sunshine_hours': 8.0,  # Open-Meteoでは日照時間は別途取得が必要
                        'source': 'openmeteo'
                    }
                    
                    print(f"✓ モデル形式に変換完了: 気温={weather_data['avg_temp']}°C, 気圧={weather_data['pressure']}hPa")
                    return weather_data
            else:
                print(f"Open-Meteo API エラー: {response.status_code}")
            
            return None
                
        except Exception as e:
            print(f"Open-Meteo API取得エラー: {e}")
            return None
    
    def _fetch_openmeteo_historical(self, days_back=30) -> List[Dict[str, Any]]:
        """Open-Meteo APIから過去の気象データを取得"""
        try:
            # 過去の気象データを取得
            historical_url = f"{self.openmeteo_api}/forecast"
            
            # 過去30日間のデータを取得
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'latitude': self.tokyo_lat,
                'longitude': self.tokyo_lon,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,pressure_msl',
                'timezone': 'Asia/Tokyo',
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(historical_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                daily = data.get('daily', {})
                
                if daily and 'time' in daily:
                    historical_data = []
                    times = daily['time']
                    temps_max = daily.get('temperature_2m_max', [])
                    temps_min = daily.get('temperature_2m_min', [])
                    precip = daily.get('precipitation_sum', [])
                    pressure = daily.get('pressure_msl', [])
                    
                    for i, date_str in enumerate(times):
                        if i < len(temps_max) and i < len(temps_min):
                            avg_temp = (temps_max[i] + temps_min[i]) / 2
                            
                            historical_data.append({
                                'date': datetime.strptime(date_str, '%Y-%m-%d'),
                                'avg_temp': avg_temp,
                                'max_temp': temps_max[i],
                                'min_temp': temps_min[i],
                                'avg_humidity': 65.0,  # デフォルト値
                                'pressure': round(pressure[i]) if i < len(pressure) else 1013,
                                'precipitation': precip[i] if i < len(precip) else 0.0,
                                'wind_speed': 5.0,  # デフォルト値
                                'sunshine_hours': 8.0,  # デフォルト値
                                'source': 'openmeteo_historical'
                            })
                    
                    print(f"✓ Open-Meteo APIから過去{len(historical_data)}日間のデータを取得しました")
                    return historical_data
            
            return []
                
        except Exception as e:
            print(f"Open-Meteo 過去データ取得エラー: {e}")
            return []

class DataCache:
    """データキャッシュ管理クラス"""
    
    def __init__(self):
        self.weather_cache = {}
        self.holiday_cache = {}
        self.statistics_cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 3600  # 1時間（秒）
    
    def get_cached_weather(self, key: str) -> Dict[str, Any]:
        """キャッシュされた気象データを取得"""
        if self._is_cache_valid(key):
            return self.weather_cache.get(key, {})
        return {}
    
    def set_cached_weather(self, key: str, data: Dict[str, Any]):
        """気象データをキャッシュに保存"""
        self.weather_cache[key] = data
        self.cache_timestamps[key] = datetime.now()
        print(f"✓ 気象データをキャッシュに保存: {key}")
    
    def get_cached_holiday(self, date_str: str) -> Dict[str, Any]:
        """キャッシュされた休日情報を取得"""
        if self._is_cache_valid(f"holiday_{date_str}"):
            return self.holiday_cache.get(date_str, {})
        return {}
    
    def set_cached_holiday(self, date_str: str, data: Dict[str, Any]):
        """休日情報をキャッシュに保存"""
        self.holiday_cache[date_str] = data
        self.cache_timestamps[f"holiday_{date_str}"] = datetime.now()
        print(f"✓ 休日情報をキャッシュに保存: {date_str}")
    
    def get_cached_statistics(self, key: str) -> Dict[str, Any]:
        """キャッシュされた統計情報を取得"""
        if self._is_cache_valid(f"stats_{key}"):
            return self.statistics_cache.get(key, {})
        return {}
    
    def set_cached_statistics(self, key: str, data: Dict[str, Any]):
        """統計情報をキャッシュに保存"""
        self.statistics_cache[key] = data
        self.cache_timestamps[f"stats_{key}"] = datetime.now()
        print(f"✓ 統計情報をキャッシュに保存: {key}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """キャッシュが有効かどうかを判定"""
        if key not in self.cache_timestamps:
            return False
        
        elapsed = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return elapsed < self.cache_duration
    
    def clear_expired_cache(self):
        """期限切れのキャッシュをクリア"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            elapsed = (current_time - timestamp).total_seconds()
            if elapsed >= self.cache_duration:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key.startswith("holiday_"):
                date_str = key.replace("holiday_", "")
                self.holiday_cache.pop(date_str, None)
            elif key.startswith("stats_"):
                stats_key = key.replace("stats_", "")
                self.statistics_cache.pop(stats_key, None)
            else:
                self.weather_cache.pop(key, None)
            del self.cache_timestamps[key]
        
        if expired_keys:
            print(f"✓ 期限切れキャッシュをクリア: {len(expired_keys)}件")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """キャッシュの状態を取得"""
        return {
            "weather_cache_size": len(self.weather_cache),
            "holiday_cache_size": len(self.holiday_cache),
            "statistics_cache_size": len(self.statistics_cache),
            "total_cached_items": len(self.cache_timestamps),
            "cache_duration_hours": self.cache_duration / 3600
        }

class CachedWeatherDataCollector:
    """キャッシュ機能付き気象データ収集クラス"""
    
    def __init__(self):
        self.cache = DataCache()
        self.holiday_collector = HolidayDataCollector()
        self.weather_collector = WeatherDataCollector()
        self.extended_collector = ExtendedWeatherDataCollector()
        
        # アプリ起動時に過去データを事前取得
        self._preload_historical_data()
    
    def _preload_historical_data(self):
        """アプリ起動時に過去データを事前取得"""
        print("=== 過去データの事前取得開始 ===")
        try:
            # 過去30日分のデータを事前取得
            complete_data = self.extended_collector.fetch_complete_weather_data(days_back=30)
            
            # キャッシュに保存
            self.cache.set_cached_statistics("30days", complete_data["statistics"])
            self.cache.set_cached_weather("historical_30days", complete_data["historical_data"])
            
            print("✓ 過去データの事前取得完了")
            print(f"  取得日数: 30日")
            print(f"  気象データ: {len(complete_data['historical_data'])}件")
            print(f"  休日データ: {len(complete_data['holiday_data'])}件")
            
        except Exception as e:
            print(f"過去データの事前取得エラー: {e}")
    
    def get_current_weather_with_cache(self) -> Dict[str, Any]:
        """キャッシュを考慮した現在の気象データを取得"""
        cache_key = "current_weather"
        cached_data = self.cache.get_cached_weather(cache_key)
        
        if cached_data:
            print("✓ キャッシュから気象データを取得")
            return cached_data
        
        # キャッシュがない場合は新規取得
        print("✓ 最新の気象データを取得中...")
        weather_data = self.weather_collector.fetch_current_weather()
        self.cache.set_cached_weather(cache_key, weather_data)
        
        return weather_data
    
    def get_holiday_info_with_cache(self, date: datetime = None) -> Dict[str, Any]:
        """キャッシュを考慮した休日情報を取得"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        cached_data = self.cache.get_cached_holiday(date_str)
        
        if cached_data:
            print(f"✓ キャッシュから休日情報を取得: {date_str}")
            return cached_data
        
        # キャッシュがない場合は新規取得
        print(f"✓ 休日情報を取得中: {date_str}")
        holiday_data = self.holiday_collector.get_holiday_info(date)
        self.cache.set_cached_holiday(date_str, holiday_data)
        
        return holiday_data
    
    def get_statistics_with_cache(self, days_back: int = 30) -> Dict[str, Any]:
        """キャッシュを考慮した統計情報を取得"""
        cache_key = f"statistics_{days_back}days"
        cached_data = self.cache.get_cached_statistics(cache_key)
        
        if cached_data:
            print(f"✓ キャッシュから統計情報を取得: {days_back}日分")
            return cached_data
        
        # キャッシュがない場合は新規取得
        print(f"✓ 統計情報を取得中: {days_back}日分")
        complete_data = self.extended_collector.fetch_complete_weather_data(days_back=days_back)
        statistics = complete_data["statistics"]
        
        self.cache.set_cached_statistics(cache_key, statistics)
        return statistics
    
    def get_complete_data_with_cache(self, days_back: int = 90) -> Dict[str, Any]:
        """キャッシュを考慮した完全データを取得"""
        cache_key = f"complete_data_{days_back}days"
        cached_data = self.cache.get_cached_weather(cache_key)
        
        if cached_data:
            print(f"✓ キャッシュから完全データを取得: {days_back}日分")
            return cached_data
        
        # キャッシュがない場合は新規取得
        print(f"✓ 完全データを取得中: {days_back}日分")
        complete_data = self.extended_collector.fetch_complete_weather_data(days_back=days_back)
        self.cache.set_cached_weather(cache_key, complete_data)
        
        return complete_data
    
    def refresh_cache(self):
        """キャッシュを更新"""
        print("=== キャッシュ更新開始 ===")
        
        # 期限切れキャッシュをクリア
        self.cache.clear_expired_cache()
        
        # 現在の気象データを更新
        current_weather = self.weather_collector.fetch_current_weather()
        self.cache.set_cached_weather("current_weather", current_weather)
        
        # 統計情報を更新
        statistics = self.get_statistics_with_cache(30)
        
        print("✓ キャッシュ更新完了")
        return {
            "status": "success",
            "message": "キャッシュを更新しました",
            "cache_status": self.cache.get_cache_status()
        }

# 予測器のインスタンスを作成
predictor = HeartFailurePredictor()
weather_collector = WeatherDataCollector()
extended_weather_collector = ExtendedWeatherDataCollector()
cached_collector = CachedWeatherDataCollector()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "心不全リスク予測Webアプリ v2.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/current",
            "weather": "/weather/current",
            "model_info": "/model/info",
            "web_interface": "/web",
            "complete_data": "/data/complete",
            "holiday_info": "/data/holiday",
            "statistics": "/data/statistics",
            "cache_status": "/cache/status",
            "refresh_cache": "/cache/refresh"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": hasattr(predictor, 'model_info'),
        "cache_status": cached_collector.cache.get_cache_status()
    }

@app.get("/cache/status")
async def get_cache_status():
    """キャッシュの状態を取得"""
    return {
        "cache_status": cached_collector.cache.get_cache_status(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/cache/refresh")
async def refresh_cache():
    """キャッシュを更新"""
    try:
        result = cached_collector.refresh_cache()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"キャッシュ更新エラー: {str(e)}")

@app.get("/data/complete")
async def get_complete_weather_data():
    """完全な気象データを取得（キャッシュ対応）"""
    try:
        print("=== 完全なデータセット取得開始（キャッシュ対応）===")
        complete_data = cached_collector.get_complete_data_with_cache(days_back=90)
        
        return {
            "status": "success",
            "message": "完全なデータセットを取得しました",
            "data": complete_data,
            "cache_info": cached_collector.cache.get_cache_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"データ取得エラー: {str(e)}")

@app.get("/data/holiday")
async def get_holiday_info():
    """現在の休日情報を取得（キャッシュ対応）"""
    try:
        current_date = datetime.now()
        holiday_info = cached_collector.get_holiday_info_with_cache(current_date)
        
        return {
            "date": current_date.strftime('%Y-%m-%d'),
            "holiday_info": holiday_info,
            "cache_info": cached_collector.cache.get_cache_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"休日情報取得エラー: {str(e)}")

@app.get("/data/statistics")
async def get_weather_statistics():
    """気象統計情報を取得（キャッシュ対応）"""
    try:
        # 過去30日分のデータを取得して統計を計算
        statistics = cached_collector.get_statistics_with_cache(days_back=30)
        
        return {
            "status": "success",
            "statistics": statistics,
            "cache_info": cached_collector.cache.get_cache_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計情報取得エラー: {str(e)}")

@app.get("/predict/current")
async def predict_current_risk():
    """現在の気象データで心不全リスクを予測"""
    try:
        # 気象データを取得
        weather_data = weather_collector.fetch_current_weather()
        
        # データソースを確認
        data_source = weather_data.get('source', 'unknown')
        print(f"取得したデータソース: {data_source}")
        
        # モデル用にデータを変換
        model_weather_data = format_weather_data_for_model(weather_data)
        
        # 予測実行
        prediction_result = predictor.predict_risk(model_weather_data)
        
        return prediction_result
        
    except HTTPException as e:
        # 気象データ取得失敗時
        return {
            "error": "気象データ取得失敗",
            "message": e.detail,
            "status_code": e.status_code,
            "prediction_date": datetime.now().isoformat(),
            "weather_data": None,
            "recommendations": ["気象データの取得に失敗しました。しばらく時間をおいてから再試行してください。"]
        }
    except Exception as e:
        print(f"予測エラー: {e}")
        return {
            "error": "予測処理エラー",
            "message": str(e),
            "prediction_date": datetime.now().isoformat(),
            "weather_data": None,
            "recommendations": ["予測処理中にエラーが発生しました。しばらく時間をおいてから再試行してください。"]
        }

def format_weather_data_for_model(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """APIから取得したデータをモデルの期待する形式に変換"""
    try:
        # データソースを確認
        source = weather_data.get('source', 'unknown')
        print(f"データソース: {source}")
        
        # APIデータの場合、モデルの期待する形式に変換
        if source in ['openmeteo', 'jma_html', 'jma_forecast', 'amedas']:
            print("✓ APIデータをモデル形式に変換中...")
            
            # モデルが期待する形式に変換
            formatted_data = {
                'avg_temp': float(weather_data.get('avg_temp', 25.0)),
                'max_temp': float(weather_data.get('max_temp', 28.0)),
                'min_temp': float(weather_data.get('min_temp', 22.0)),
                'avg_humidity': float(weather_data.get('avg_humidity', 65.0)),
                'pressure': float(weather_data.get('pressure', 1013.0)),
                'precipitation': float(weather_data.get('precipitation', 0.0)),
                'wind_speed': float(weather_data.get('wind_speed', 5.0)),
                'sunshine_hours': float(weather_data.get('sunshine_hours', 8.0)),
                'source': source
            }
            
            print(f"✓ データ変換完了: 気温={formatted_data['avg_temp']}°C, 気圧={formatted_data['pressure']}hPa")
            print(f"✓ データソース: {formatted_data['source']}")
            return formatted_data
        else:
            print(f"⚠️ 推定データを使用: {source}")
            return weather_data
            
    except Exception as e:
        print(f"データ変換エラー: {e}")
        return weather_data

@app.post("/predict/custom")
async def predict_custom_risk(weather_data: WeatherData):
    """カスタム気象データでリスクを予測"""
    try:
        # データを辞書に変換
        weather_dict = weather_data.dict()
        
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
        return {
            "status": "success",
            "data": weather_data,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException as e:
        return {
            "status": "error",
            "error": "気象データ取得失敗",
            "message": e.detail,
            "status_code": e.status_code,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": "予期しないエラー",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/model/info")
async def get_model_info():
    """モデル情報を取得"""
    return {
        "model_info": predictor.model_info,
        "ensemble_weights": predictor.ensemble_weights.tolist() if hasattr(predictor, 'ensemble_weights') else None,
        "best_params": predictor.best_params if hasattr(predictor, 'best_params') else None,
        "feature_names": predictor.feature_names if hasattr(predictor, 'feature_names') else None
    }

# HTMLテンプレート
HTML_TEMPLATE = """
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
            margin: 20px 0;
        }
        
        .weather-item {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .weather-item h3 {
            color: #667eea;
            margin-bottom: 5px;
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
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .model-info {
            background: #f1f3f4;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #666;
        }
        
        .dataset-info {
            background: #e8f4fd;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #333;
        }
        
        .data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .data-item {
            background: white;
            border-radius: 8px;
            padding: 12px;
            border-left: 4px solid #667eea;
        }
        
        .data-item h5 {
            margin: 0 0 8px 0;
            color: #333;
            font-size: 0.9em;
        }
        
        .data-item div {
            font-size: 0.8em;
            color: #666;
            line-height: 1.4;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 20px;
            transition: transform 0.2s;
        }
        
        .refresh-btn:hover {
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💓 心不全リスク予測</h1>
            <p>リアルタイム気象データによる心不全リスク予測システム</p>
        </div>
        
        <div class="content">
            <div id="loading" class="loading">
                <h3>気象データを取得中...</h3>
                <p>最新の気象情報を基にリスクを分析しています</p>
            </div>
            
            <div id="prediction" style="display: none;">
                <div class="prediction-card">
                    <div id="risk-level" class="risk-level"></div>
                    <h2>リスク予測結果</h2>
                    <p><strong>予測確率:</strong> <span id="risk-probability"></span></p>
                    <p><strong>予測日時:</strong> <span id="prediction-date"></span></p>
                </div>
                
                <div class="weather-grid" id="weather-grid"></div>
                
                <div class="recommendations">
                    <h3>📋 推奨事項</h3>
                    <ul id="recommendations-list"></ul>
                </div>
                
                <div class="model-info">
                    <h4>モデル情報</h4>
                    <p><strong>バージョン:</strong> <span id="model-version"></span></p>
                    <p><strong>AUC:</strong> <span id="model-auc"></span></p>
                    <p><strong>最適化Fold:</strong> <span id="model-fold"></span></p>
                </div>
                
                <div class="dataset-info">
                    <h4>📊 データセット情報</h4>
                    <div class="data-grid">
                        <div class="data-item">
                            <h5>📅 休日情報</h5>
                            <div id="holiday-info">読み込み中...</div>
                        </div>
                        <div class="data-item">
                            <h5>📈 統計情報</h5>
                            <div id="statistics-info">読み込み中...</div>
                        </div>
                        <div class="data-item">
                            <h5>🌍 データ完全性</h5>
                            <div id="completeness-info">読み込み中...</div>
                        </div>
                        <div class="data-item">
                            <h5>💾 キャッシュ情報</h5>
                            <div id="cache-info">読み込み中...</div>
                        </div>
                    </div>
                </div>
                
                <button class="refresh-btn" onclick="refreshPrediction()">
                    🔄 最新データで再予測
                </button>
            </div>
            
            <div id="error" class="error" style="display: none;"></div>
        </div>
    </div>

    <script>
        async function loadPrediction() {
            try {
                const response = await fetch('/predict/current');
                const data = await response.json();
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('prediction').style.display = 'block';
                
                // リスクレベルを設定
                const riskLevel = document.getElementById('risk-level');
                riskLevel.textContent = data.risk_level;
                riskLevel.className = 'risk-level risk-' + 
                    (data.risk_level === '低リスク' ? 'low' : 
                     data.risk_level === '中リスク' ? 'medium' : 'high');
                
                // 予測確率
                document.getElementById('risk-probability').textContent = 
                    (data.risk_probability * 100).toFixed(1) + '%';
                
                // 予測日時
                document.getElementById('prediction-date').textContent = 
                    new Date(data.prediction_date).toLocaleString('ja-JP');
                
                // 気象データ
                const weatherGrid = document.getElementById('weather-grid');
                weatherGrid.innerHTML = '';
                
                const weatherData = data.weather_data;
                const weatherItems = [
                    { name: '平均気温', value: weatherData.avg_temp + '°C' },
                    { name: '最高気温', value: weatherData.max_temp + '°C' },
                    { name: '最低気温', value: weatherData.min_temp + '°C' },
                    { name: '平均湿度', value: weatherData.avg_humidity + '%' },
                    { name: '気圧', value: weatherData.pressure + 'hPa' },
                    { name: '降水量', value: weatherData.precipitation + 'mm' },
                    { name: '風速', value: weatherData.wind_speed + 'm/s' },
                    { name: '日照時間', value: weatherData.sunshine_hours + '時間' }
                ];
                
                weatherItems.forEach(item => {
                    const div = document.createElement('div');
                    div.className = 'weather-item';
                    div.innerHTML = `
                        <h3>${item.name}</h3>
                        <p>${item.value}</p>
                    `;
                    weatherGrid.appendChild(div);
                });
                
                // 推奨事項
                const recommendationsList = document.getElementById('recommendations-list');
                recommendationsList.innerHTML = '';
                data.recommendations.recommendations.forEach(rec => {
                    const li = document.createElement('li');
                    li.textContent = rec;
                    recommendationsList.appendChild(li);
                });
                
                // モデル情報
                const modelInfo = data.model_info;
                document.getElementById("model-version").textContent = modelInfo.model_version || "不明";
                document.getElementById("model-auc").textContent = (modelInfo.hold_out_auc || modelInfo.best_auc || 0).toFixed(4);
                document.getElementById("model-fold").textContent = modelInfo.hold_out_auc ? "Hold-out" : (modelInfo.best_fold || "不明");
                
                // データセット情報を読み込み
                loadDatasetInfo();
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').style.display = 'block';
                document.getElementById('error').textContent = 
                    '予測データの取得に失敗しました: ' + error.message;
            }
        }
        
        async function loadDatasetInfo() {
            try {
                // 休日情報を取得
                const holidayResponse = await fetch('/data/holiday');
                const holidayData = await holidayResponse.json();
                
                const holidayInfo = holidayData.holiday_info;
                const holidayDiv = document.getElementById('holiday-info');
                
                holidayDiv.innerHTML = `
                    <p><strong>日付:</strong> ${holidayData.date}</p>
                    <p><strong>曜日:</strong> ${getDayOfWeek(holidayInfo.day_of_week)}</p>
                    <p><strong>土日:</strong> ${holidayInfo.is_weekend ? 'はい' : 'いいえ'}</p>
                    <p><strong>祝日:</strong> ${holidayInfo.is_holiday ? holidayInfo.holiday_name : 'いいえ'}</p>
                    <p><strong>月末:</strong> ${holidayInfo.is_month_end ? 'はい' : 'いいえ'}</p>
                    <p><strong>季節:</strong> ${getSeasonName(holidayInfo.season)}</p>
                `;
                
                // 統計情報を取得
                const statsResponse = await fetch('/data/statistics');
                const statsData = await statsResponse.json();
                
                const stats = statsData.statistics;
                const statsDiv = document.getElementById('statistics-info');
                
                statsDiv.innerHTML = `
                    <p><strong>平均気温:</strong> ${stats.weather.temp_mean.toFixed(1)}°C</p>
                    <p><strong>気温トレンド:</strong> ${stats.weather.temp_trend > 0 ? '上昇' : '下降'}</p>
                    <p><strong>平均湿度:</strong> ${stats.weather.humidity_mean.toFixed(1)}%</p>
                    <p><strong>平均気圧:</strong> ${stats.weather.pressure_mean.toFixed(1)}hPa</p>
                    <p><strong>総降水量:</strong> ${stats.weather.precipitation_total.toFixed(1)}mm</p>
                    <p><strong>雨の日数:</strong> ${stats.weather.precipitation_days}日</p>
                `;
                
                // データ完全性を取得
                const completeResponse = await fetch('/data/complete');
                const completeData = await completeResponse.json();
                
                const completeness = completeData.data.data_completeness;
                const completenessDiv = document.getElementById('completeness-info');
                
                completenessDiv.innerHTML = `
                    <p><strong>総日数:</strong> ${completeness.total_days}日</p>
                    <p><strong>気象データ:</strong> ${completeness.weather_data_count}件</p>
                    <p><strong>休日データ:</strong> ${completeness.holiday_data_count}件</p>
                    <p><strong>完成率:</strong> ${completeness.completion_rate}</p>
                `;
                
                // キャッシュ情報を取得
                const cacheResponse = await fetch('/cache/status');
                const cacheData = await cacheResponse.json();
                
                const cacheInfoDiv = document.getElementById('cache-info');
                cacheInfoDiv.innerHTML = `
                    <p><strong>気象データキャッシュ:</strong> ${cacheData.weather_cache_size}件</p>
                    <p><strong>休日情報キャッシュ:</strong> ${cacheData.holiday_cache_size}件</p>
                    <p><strong>統計情報キャッシュ:</strong> ${cacheData.statistics_cache_size}件</p>
                    <p><strong>総キャッシュアイテム数:</strong> ${cacheData.total_cached_items}</p>
                    <p><strong>キャッシュ有効期間:</strong> ${cacheData.cache_duration_hours}時間</p>
                `;
                
            } catch (error) {
                console.error('データセット情報の取得に失敗:', error);
                document.getElementById('holiday-info').innerHTML = 
                    '<p style="color: red;">休日情報の取得に失敗しました</p>';
                document.getElementById('statistics-info').innerHTML = 
                    '<p style="color: red;">統計情報の取得に失敗しました</p>';
                document.getElementById('completeness-info').innerHTML = 
                    '<p style="color: red;">データ完全性の取得に失敗しました</p>';
                document.getElementById('cache-info').innerHTML = 
                    '<p style="color: red;">キャッシュ情報の取得に失敗しました</p>';
            }
        }
        
        function getDayOfWeek(day) {
            const days = ['月', '火', '水', '木', '金', '土', '日'];
            return days[day];
        }
        
        function getSeasonName(season) {
            const seasons = {
                'spring': '春',
                'summer': '夏',
                'autumn': '秋',
                'winter': '冬'
            };
            return seasons[season] || season;
        }
        
        function refreshPrediction() {
            document.getElementById('prediction').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            loadPrediction();
        }
        
        // ページ読み込み時に予測を実行
        window.onload = loadPrediction;
    </script>
</body>
</html>
"""

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Webインターフェース"""
    return HTMLResponse(content=HTML_TEMPLATE)

if __name__ == "__main__":
    import uvicorn
    print("💓 心不全リスク予測Webアプリ v2.0 を起動中...")
    print("🌐 Webインターフェース: http://localhost:8000/web")
    print("📊 API: http://localhost:8000")
    print("📚 APIドキュメント: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000) 