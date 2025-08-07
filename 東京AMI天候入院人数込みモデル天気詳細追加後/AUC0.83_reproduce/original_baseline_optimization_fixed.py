#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オリジナルベースライン（AUC 0.8348）最適化システム - 修正版
時系列分割 + 異常気象フラグ + 暴風警フラグ + ハイパーパラメータ自動調整 + NaN除去
"""

import pandas as pd
import numpy as np
import warnings
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import logging

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OriginalBaselineOptimizer:
    def __init__(self, target_auc=0.90, max_time_hours=8):
        """
        オリジナルベースライン最適化システム
        
        Args:
            target_auc: 目標AUC（デフォルト: 0.90）
            max_time_hours: 最大実行時間（時間）
        """
        self.target_auc = target_auc
        self.max_time_hours = max_time_hours
        self.start_time = time.time()
        self.baseline_auc = 0.8348  # オリジナルモデルのAUC
        
        # 結果保存用
        self.best_auc = self.baseline_auc
        self.best_model = None
        self.best_features = None
        self.optimization_history = []
        
        logger.info(f"🎯 オリジナルベースライン最適化開始")
        logger.info(f"ベースAUC: {self.baseline_auc:.4f}")
        logger.info(f"目標AUC: {self.target_auc:.4f}")
        logger.info(f"最大実行時間: {self.max_time_hours}時間")
    
    def load_original_data(self):
        """オリジナルデータの読み込み（2012-2019年、コロナ禍除外）"""
        try:
            # オリジナルデータファイルのパス
            file_path = '../東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv'
            
            logger.info("📊 オリジナルデータ読み込み中...")
            df = pd.read_csv(file_path)
            
            # オリジナル期間（2012-2019年）に絞り込み
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= '2012-04-01') & (df['date'] <= '2019-12-31')]
            
            # カラム名を統一（people → hospitalization_count）
            if 'people' in df.columns:
                df['hospitalization_count'] = df['people']
            
            logger.info(f"データ形状: {df.shape}")
            logger.info(f"期間: {df['date'].min()} から {df['date'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return None
    
    def create_original_features(self, df):
        """オリジナルモデルと同じ特徴量を作成"""
        logger.info("🔧 オリジナル特徴量作成中...")
        
        # 基本時間特徴量
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # 季節性特徴量
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 祝日・休日フラグ
        df['is_holiday'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # 気象変化特徴量（実際のカラム名に合わせて修正）
        for col in ['avg_temp', 'avg_humidity', 'pressure_local']:
            if col in df.columns:
                df[f'{col}_change'] = df[col].diff()
                df[f'{col}_change_3d'] = df[col].diff(3)
                df[f'{col}_change_7d'] = df[col].diff(7)
        
        # 気温変動パターン
        if 'avg_temp' in df.columns:
            df['temp_range'] = df['max_temp'] - df['min_temp']
            df['temp_change_5deg'] = (df['avg_temp'].diff().abs() >= 5).astype(int)
        
        # 不快指数
        if all(col in df.columns for col in ['avg_temp', 'avg_humidity']):
            df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
        
        # 熱帯夜フラグ
        if 'min_temp' in df.columns:
            df['tropical_night'] = (df['min_temp'] >= 25).astype(int)
        
        # 猛暑日フラグ
        if 'max_temp' in df.columns:
            df['extremely_hot'] = (df['max_temp'] >= 35).astype(int)
        
        # 真夏日フラグ
        if 'max_temp' in df.columns:
            df['summer_day'] = (df['max_temp'] >= 30).astype(int)
        
        # 冬日フラグ
        if 'min_temp' in df.columns:
            df['winter_day'] = (df['min_temp'] < 0).astype(int)
        
        # 強風フラグ
        if 'avg_wind' in df.columns:
            df['strong_wind'] = (df['avg_wind'] >= 10).astype(int)
        
        # 入院データのラグ特徴量
        if 'hospitalization_count' in df.columns:
            for lag in [1, 2, 3, 7, 14, 28]:
                df[f'hospitalization_lag_{lag}'] = df['hospitalization_count'].shift(lag)
            
            # 移動平均
            for window in [7, 14, 28]:
                df[f'hospitalization_ma_{window}'] = df['hospitalization_count'].rolling(window=window).mean()
                df[f'hospitalization_std_{window}'] = df['hospitalization_count'].rolling(window=window).std()
        
        logger.info(f"特徴量作成完了: {df.shape[1]}列")
        return df
    
    def create_enhanced_weather_flags(self, df):
        """異常気象フラグと暴風警フラグの追加作成"""
        logger.info("🌪️ 異常気象フラグ・暴風警フラグ作成中...")
        
        # 台風・暴風系フラグ
        if 'avg_wind' in df.columns:
            df['typhoon_wind'] = (df['avg_wind'] >= 15).astype(int)
            df['storm_wind'] = (df['avg_wind'] >= 20).astype(int)
            df['hurricane_wind'] = (df['avg_wind'] >= 25).astype(int)
        
        # 急激な降雨・雷雨系フラグ
        if 'precipitation' in df.columns:
            df['heavy_rain'] = (df['precipitation'] >= 50).astype(int)
            df['torrential_rain'] = (df['precipitation'] >= 100).astype(int)
            df['rain_change'] = df['precipitation'].diff()
            df['sudden_rain'] = (df['rain_change'] >= 30).astype(int)
        
        # 急激な気温変化系フラグ
        if 'avg_temp' in df.columns:
            temp_change = df['avg_temp'].diff()
            df['temp_drop_5deg'] = (temp_change <= -5).astype(int)
            df['temp_drop_10deg'] = (temp_change <= -10).astype(int)
            df['temp_rise_5deg'] = (temp_change >= 5).astype(int)
            df['temp_rise_10deg'] = (temp_change >= 10).astype(int)
        
        # 極端な高温・熱帯化フラグ
        if 'max_temp' in df.columns:
            df['heat_wave'] = (df['max_temp'] >= 35).astype(int)
            df['extreme_heat'] = (df['max_temp'] >= 38).astype(int)
            df['record_heat'] = (df['max_temp'] >= 40).astype(int)
        
        # 極端な低温・寒波フラグ
        if 'min_temp' in df.columns:
            df['cold_wave'] = (df['min_temp'] <= -5).astype(int)
            df['extreme_cold'] = (df['min_temp'] <= -10).astype(int)
            df['record_cold'] = (df['min_temp'] <= -15).astype(int)
        
        # 乾燥・低湿度ストレスフラグ
        if 'avg_humidity' in df.columns:
            df['dry_weather'] = (df['avg_humidity'] <= 30).astype(int)
            df['very_dry'] = (df['avg_humidity'] <= 20).astype(int)
            df['extremely_dry'] = (df['avg_humidity'] <= 10).astype(int)
        
        # 気圧変動ストレスフラグ
        if 'pressure_local' in df.columns:
            pressure_change = df['pressure_local'].diff()
            df['pressure_drop'] = (pressure_change <= -5).astype(int)
            df['pressure_rise'] = (pressure_change >= 5).astype(int)
            df['pressure_volatile'] = (pressure_change.abs() >= 10).astype(int)
        
        # 複合異常気象フラグ
        if all(col in df.columns for col in ['max_temp', 'avg_humidity', 'avg_wind']):
            df['heat_humidity_stress'] = ((df['max_temp'] >= 30) & (df['avg_humidity'] >= 70)).astype(int)
            df['cold_wind_stress'] = ((df['min_temp'] <= 0) & (df['avg_wind'] >= 5)).astype(int)
            df['dry_wind_stress'] = ((df['avg_humidity'] <= 30) & (df['avg_wind'] >= 8)).astype(int)
        
        # 異常気象フラグのラグ特徴量
        weather_flags = [col for col in df.columns if any(keyword in col for keyword in 
                        ['typhoon', 'storm', 'hurricane', 'heavy_rain', 'torrential_rain', 
                         'sudden_rain', 'temp_drop', 'temp_rise', 'heat_wave', 'extreme_heat',
                         'record_heat', 'cold_wave', 'extreme_cold', 'record_cold', 'dry_weather',
                         'very_dry', 'extremely_dry', 'pressure_drop', 'pressure_rise', 'pressure_volatile',
                         'heat_humidity_stress', 'cold_wind_stress', 'dry_wind_stress'])]
        
        for flag in weather_flags:
            for lag in [1, 2, 3]:
                df[f'{flag}_lag_{lag}'] = df[flag].shift(lag)
        
        logger.info(f"異常気象フラグ作成完了: {len(weather_flags)}個のフラグ")
        return df
    
    def create_target_variable(self, df, threshold_percentile=80):
        """ターゲット変数の作成（オリジナルと同じ方法）"""
        logger.info("🎯 ターゲット変数作成中...")
        
        if 'hospitalization_count' not in df.columns:
            logger.error("入院データが見つかりません")
            return None
        
        # 80パーセンタイルを閾値として使用
        threshold = df['hospitalization_count'].quantile(threshold_percentile / 100)
        df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
        
        logger.info(f"閾値: {threshold:.1f}人")
        logger.info(f"高リスク日割合: {df['target'].mean():.3f}")
        
        return df
    
    def time_series_split_validation(self, X, y, n_splits=5):
        """時系列分割による検証"""
        logger.info("⏰ 時系列分割検証実行中...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * 0.2))
        
        # 複数モデルでの時系列検証
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'CatBoost': cb.CatBoostClassifier(iterations=100, random_state=42, verbose=False)
        }
        
        results = {}
        for name, model in models.items():
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                scores.append(auc)
            
            mean_auc = np.mean(scores)
            std_auc = np.std(scores)
            results[name] = {'mean_auc': mean_auc, 'std_auc': std_auc}
            
            logger.info(f"  {name}: AUC {mean_auc:.4f} (±{std_auc:.4f})")
        
        # 最良モデルを選択
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean_auc'])
        best_auc = results[best_model_name]['mean_auc']
        
        logger.info(f"最良モデル: {best_model_name} (AUC: {best_auc:.4f})")
        return best_model_name, best_auc, results
    
    def run_optimization(self):
        """メイン最適化プロセス"""
        logger.info("🚀 オリジナルベースライン最適化開始")
        
        # 1. データ読み込み
        df = self.load_original_data()
        if df is None:
            return None, 0.0
        
        # 2. オリジナル特徴量作成
        df = self.create_original_features(df)
        
        # 3. 異常気象フラグ追加
        df = self.create_enhanced_weather_flags(df)
        
        # 4. ターゲット変数作成
        df = self.create_target_variable(df)
        if df is None:
            return None, 0.0
        
        # 5. 特徴量とターゲットの準備
        # 数値カラムのみを特徴量として選択
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in ['hospitalization_count', 'target']]
        X = df[feature_columns]
        y = df['target']
        
        # NaNの完全除去
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"最終データ形状: X={X.shape}, y={y.shape}")
        logger.info(f"特徴量数: {len(feature_columns)}")
        logger.info(f"除外された文字列カラム: {[col for col in df.columns if col not in numeric_columns + ['date', 'hospitalization_count', 'target']]}")
        logger.info(f"NaNチェック: XにNaN {X.isnull().sum().sum()}個")
        
        # 6. 時系列分割検証
        best_model_name, baseline_auc, validation_results = self.time_series_split_validation(X, y)
        
        logger.info(f"🏆 ベースラインAUC: {baseline_auc:.4f}")
        logger.info(f"オリジナルAUC: {self.baseline_auc:.4f}")
        logger.info(f"改善: {baseline_auc - self.baseline_auc:.4f}")
        
        return None, baseline_auc

def main():
    """メイン実行関数"""
    print("🎯 オリジナルベースライン最適化システム - 修正版")
    print("=" * 60)
    
    # 最適化システムの初期化
    optimizer = OriginalBaselineOptimizer(
        target_auc=0.90,  # 目標AUC
        max_time_hours=8  # 最大実行時間
    )
    
    # 最適化実行
    best_model, best_auc = optimizer.run_optimization()
    
    if best_model is not None:
        print(f"\n🎉 最適化完了！")
        print(f"最終AUC: {best_auc:.4f}")
    else:
        print(f"\n✅ ベースライン確認完了！")
        print(f"ベースラインAUC: {best_auc:.4f}")

if __name__ == "__main__":
    main() 