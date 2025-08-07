
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class HeartFailureWeatherPredictor:
    """心不全気象予測モデル"""
    
    def __init__(self, model_dir):
        """モデルを読み込み"""
        self.model_dir = model_dir
        
        # モデル情報を読み込み
        with open(f'{model_dir}/model_info.json', 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)
        
        # ハイパーパラメータを読み込み
        with open(f'{model_dir}/best_hyperparameters.pkl', 'rb') as f:
            self.best_params = pickle.load(f)
        
        # アンサンブル重みを読み込み
        self.ensemble_weights = np.load(f'{model_dir}/ensemble_weights.npy')
        
        # スケーラーを初期化
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """心不全特化の特徴量を作成"""
        # 基本気象データの前処理
        weather_cols = ['min_temp_weather', 'max_temp_weather', 'avg_temp_weather', 
                       'avg_wind_weather', 'pressure_local', 'avg_humidity_weather', 
                       'sunshine_hours_weather', 'precipitation']
        
        df[weather_cols] = df[weather_cols].fillna(method='ffill')
        df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
        
        # 心不全に影響する温度変化
        df['temp_range'] = df['max_temp_weather'] - df['min_temp_weather']
        df['temp_change_from_yesterday'] = df['avg_temp_weather'].diff()
        df['temp_change_3day'] = df['avg_temp_weather'].diff(3)
        
        # 心不全悪化のリスク要因
        df['is_cold_stress'] = (df['min_temp_weather'] < 5).astype(int)
        df['is_heat_stress'] = (df['max_temp_weather'] > 30).astype(int)
        df['is_temperature_shock'] = (abs(df['temp_change_from_yesterday']) > 10).astype(int)
        
        # 湿度関連
        df['is_high_humidity'] = (df['avg_humidity_weather'] > 80).astype(int)
        df['is_low_humidity'] = (df['avg_humidity_weather'] < 30).astype(int)
        df['humidity_change'] = df['avg_humidity_weather'].diff()
        
        # 気圧関連
        df['pressure_change'] = df['pressure_local'].diff()
        df['pressure_change_3day'] = df['pressure_local'].diff(3)
        df['is_pressure_drop'] = (df['pressure_change'] < -5).astype(int)
        df['is_pressure_rise'] = (df['pressure_change'] > 5).astype(int)
        
        # 風関連
        df['is_strong_wind'] = (df['avg_wind_weather'] > 10).astype(int)
        df['wind_change'] = df['avg_wind_weather'].diff()
        
        # 降水量関連
        df['is_rainy'] = (df['precipitation'] > 0).astype(int)
        
        # 日付特徴量
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week
        
        # 心不全に影響する曜日パターン
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        
        # 季節性
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # 冬期フラグ
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # 月末・月初
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        return df
    
    def predict(self, weather_data):
        """心不全悪化リスクを予測"""
        # 特徴量を作成
        df = self.create_features(weather_data)
        
        # 特徴量を選択（学習時と同じ）
        exclude_cols = ['date', 'people_hf', 'target', 'season']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 欠損値処理
        numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 特徴量を準備
        X = df[feature_columns].values
        
        # スケーリング
        X_scaled = self.scaler.fit_transform(X)
        
        # 各モデルの予測（実際のモデルファイルが必要）
        # ここでは例として重み付き平均を返す
        predictions = np.random.random(len(X))  # 実際の予測に置き換え
        
        return predictions
    
    def get_risk_level(self, prediction_prob):
        """リスクレベルを判定"""
        if prediction_prob >= 0.7:
            return "高リスク"
        elif prediction_prob >= 0.4:
            return "中リスク"
        else:
            return "低リスク"

# 使用例
if __name__ == "__main__":
    # モデルを読み込み
    predictor = HeartFailureWeatherPredictor("心不全気象予測モデル_50回Fold最適化版_保存モデル")
    
    # 気象データを準備（例）
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
    
    # 結果表示
    for i, (date, pred) in enumerate(zip(weather_data['date'], predictions)):
        risk_level = predictor.get_risk_level(pred)
        print(f"{date.strftime('%Y-%m-%d')}: リスク確率 {pred:.3f} ({risk_level})")
