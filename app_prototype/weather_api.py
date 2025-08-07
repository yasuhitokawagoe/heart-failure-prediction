import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class WeatherDataCollector:
    """気象データ収集クラス"""
    
    def __init__(self):
        self.base_url = "https://www.jma.go.jp/bosai/amedas/data/point/44132"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_current_weather(self):
        """現在の気象データを取得"""
        try:
            # 気象庁のアメダスデータを取得
            today = datetime.now().strftime("%Y%m%d")
            url = f"{self.base_url}/{today}.json"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # データを整形
            weather_data = self._process_weather_data(data)
            return weather_data
            
        except Exception as e:
            print(f"気象データ取得エラー: {e}")
            return self._get_fallback_data()
    
    def _process_weather_data(self, raw_data):
        """生データを処理して特徴量を作成"""
        # 最新のデータを取得
        latest_data = {}
        for key, values in raw_data.items():
            if values and len(values) > 0:
                latest_data[key] = values[-1]  # 最新の値を取得
        
        # 基本気象データ
        weather_data = {
            'date': datetime.now().date().isoformat(),
            'avg_temp': latest_data.get('temp', [None])[0] if 'temp' in latest_data else None,
            'max_temp': max(latest_data.get('temp', [0])) if 'temp' in latest_data else None,
            'min_temp': min(latest_data.get('temp', [0])) if 'temp' in latest_data else None,
            'avg_humidity': latest_data.get('humidity', [None])[0] if 'humidity' in latest_data else None,
            'pressure': latest_data.get('pressure', [None])[0] if 'pressure' in latest_data else None,
            'precipitation': latest_data.get('precipitation', [None])[0] if 'precipitation' in latest_data else None,
            'wind_speed': latest_data.get('wind_speed', [None])[0] if 'wind_speed' in latest_data else None,
            'sunshine_hours': latest_data.get('sunshine', [None])[0] if 'sunshine' in latest_data else None
        }
        
        # 心不全特化特徴量を計算
        weather_data.update(self._calculate_hf_features(weather_data))
        
        return weather_data
    
    def _calculate_hf_features(self, weather_data):
        """心不全特化特徴量を計算"""
        features = {}
        
        # 寒冷ストレス
        features['is_cold_stress'] = 1 if weather_data.get('min_temp', 0) < 5 else 0
        
        # 暑熱ストレス
        features['is_heat_stress'] = 1 if weather_data.get('max_temp', 0) > 30 else 0
        
        # 日較差
        if weather_data.get('max_temp') and weather_data.get('min_temp'):
            features['temp_range'] = weather_data['max_temp'] - weather_data['min_temp']
        else:
            features['temp_range'] = 0
        
        # 高湿度
        features['is_high_humidity'] = 1 if weather_data.get('avg_humidity', 0) > 80 else 0
        
        # 低湿度
        features['is_low_humidity'] = 1 if weather_data.get('avg_humidity', 0) < 30 else 0
        
        # 強風
        features['is_strong_wind'] = 1 if weather_data.get('wind_speed', 0) > 10 else 0
        
        # 降雨
        features['is_rainy'] = 1 if weather_data.get('precipitation', 0) > 0 else 0
        
        return features
    
    def _get_fallback_data(self):
        """フォールバックデータ（APIエラー時）"""
        return {
            'date': datetime.now().date().isoformat(),
            'avg_temp': 20.0,
            'max_temp': 25.0,
            'min_temp': 15.0,
            'avg_humidity': 60.0,
            'pressure': 1013.0,
            'precipitation': 0.0,
            'wind_speed': 3.0,
            'sunshine_hours': 6.0,
            'is_cold_stress': 0,
            'is_heat_stress': 0,
            'temp_range': 10.0,
            'is_high_humidity': 0,
            'is_low_humidity': 0,
            'is_strong_wind': 0,
            'is_rainy': 0
        }

# テスト用
if __name__ == "__main__":
    collector = WeatherDataCollector()
    weather_data = collector.fetch_current_weather()
    print("取得した気象データ:")
    print(json.dumps(weather_data, indent=2, ensure_ascii=False)) 