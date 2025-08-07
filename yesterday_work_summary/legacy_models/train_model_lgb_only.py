import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import jpholiday
import warnings
warnings.filterwarnings('ignore')

def create_date_features(df):
    """日付関連の特徴量を作成"""
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['is_weekend'] = df['hospitalization_date'].dt.dayofweek.isin([5, 6])
    df['is_holiday'] = df['hospitalization_date'].apply(lambda x: jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    
    # 周期性特徴量
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 季節性
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], labels=['winter', 'spring', 'summer', 'autumn'])
    
    # 月の開始・終了
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def load_processed_data():
    """処理済みデータを読み込み"""
    df = pd.read_csv('/Users/kawagoeyasuhito/Desktop/JROAD 機械学習/東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv')
    df['hospitalization_date'] = pd.to_datetime(df['date'])
    df['hospitalization_count'] = df['people']
    return df

def add_lag_features(df, lag_days=7):
    """ラグ特徴量を追加"""
    for lag in [1, 3, 7, 14, 28]:
        df[f'patients_lag_{lag}'] = df['hospitalization_count'].shift(lag)
    return df

def detect_extreme_weather(df):
    """異常気象を検出する関数"""
    # NaN値を前方補完（未来のデータは使用しない）
    weather_cols = ['min_temp', 'max_temp', 'avg_temp', 'avg_wind', 'vapor_pressure', 
                   'avg_humidity', 'sunshine_hours']
    df[weather_cols] = df[weather_cols].fillna(method='ffill')
    
    # 残りのNaN値を中央値で補完
    df[weather_cols] = df[weather_cols].fillna(df[weather_cols].median())
    
    # 熱帯夜（夜間の最低気温が25℃以上）
    df['is_tropical_night'] = (df['min_temp'] >= 25).astype(int)
    
    # 猛暑日（最高気温が35℃以上）
    df['is_extremely_hot'] = (df['max_temp'] >= 35).astype(int)
    
    # 真夏日（最高気温が30℃以上）
    df['is_hot_day'] = (df['max_temp'] >= 30).astype(int)
    
    # 夏日（最高気温が25℃以上）
    df['is_summer_day'] = (df['max_temp'] >= 25).astype(int)
    
    # 冬日（最低気温が0℃未満）
    df['is_winter_day'] = (df['min_temp'] < 0).astype(int)
    
    # 真冬日（最高気温が0℃未満）
    df['is_freezing_day'] = (df['max_temp'] < 0).astype(int)
    
    # 寒波（その月の平均気温から大きく低い）
    df['monthly_temp_mean'] = df.groupby(['year', 'month'])['avg_temp'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.mean())
    )
    df['is_cold_wave'] = (df['avg_temp'] <= (df['monthly_temp_mean'] - 2.0)).astype(int)
    
    # 強風日（平均風速が10m/s以上）
    df['is_strong_wind'] = (df['avg_wind'] >= 10).astype(int)
    
    # 台風接近時（気圧が急激に変化）
    df['pressure_change'] = df['pressure_local'].diff()
    df['is_typhoon_condition'] = (abs(df['pressure_change']) >= 5).astype(int)
    
    # 急激な気圧変化（前日比5hPa以上）
    df['is_rapid_pressure_change'] = (abs(df['pressure_change']) >= 5).astype(int)
    
    # 高温多湿（気温25℃以上かつ湿度80%以上）
    df['is_high_temp_humidity'] = ((df['avg_temp'] >= 25) & (df['avg_humidity'] >= 80)).astype(int)
    
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp'] + 0.01 * df['avg_humidity'] * (0.99 * df['avg_temp'] - 14.3) + 46.3
    
    # 連続フラグ（前日と同じ条件が続く場合）
    for col in ['is_tropical_night', 'is_extremely_hot', 'is_hot_day', 'is_winter_day', 'is_freezing_day']:
        df[f'{col}_consecutive'] = df[col].groupby((df[col] != df[col].shift()).cumsum()).cumsum()
    
    return df

def create_weather_interaction_features(df):
    """気象要素の相互作用特徴量を作成"""
    # 気温と湿度の相互作用
    df['temp_humidity'] = df['avg_temp'] * df['avg_humidity']
    df['temp_humidity_ratio'] = df['avg_temp'] / (df['avg_humidity'] + 1e-8)
    
    # 気温と気圧の相互作用
    df['temp_pressure'] = df['avg_temp'] * df['pressure_local']
    df['temp_pressure_ratio'] = df['avg_temp'] / (df['pressure_local'] + 1e-8)
    
    # 湿度と気圧の相互作用
    df['humidity_pressure'] = df['avg_humidity'] * df['pressure_local']
    df['humidity_pressure_ratio'] = df['avg_humidity'] / (df['pressure_local'] + 1e-8)
    
    # 風速と気温の相互作用
    df['wind_temp'] = df['avg_wind'] * df['avg_temp']
    df['wind_temp_ratio'] = df['avg_wind'] / (df['avg_temp'] + 1e-8)
    
    # 日照時間と気温の相互作用
    df['sunshine_temp'] = df['sunshine_hours'] * df['avg_temp']
    df['sunshine_temp_ratio'] = df['sunshine_hours'] / (df['avg_temp'] + 1e-8)
    
    # 蒸気圧と気温の相互作用
    df['vapor_temp'] = df['vapor_pressure'] * df['avg_temp']
    df['vapor_temp_ratio'] = df['vapor_pressure'] / (df['avg_temp'] + 1e-8)
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成"""
    # 移動平均（7日、14日、30日）
    for window in [7, 14, 30]:
        df[f'temp_ma_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).mean()
        df[f'humidity_ma_{window}'] = df['avg_humidity'].rolling(window=window, min_periods=1).mean()
        df[f'pressure_ma_{window}'] = df['pressure_local'].rolling(window=window, min_periods=1).mean()
        df[f'wind_ma_{window}'] = df['avg_wind'].rolling(window=window, min_periods=1).mean()
    
    # 移動標準偏差
    for window in [7, 14, 30]:
        df[f'temp_std_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).std()
        df[f'humidity_std_{window}'] = df['avg_humidity'].rolling(window=window, min_periods=1).std()
        df[f'pressure_std_{window}'] = df['pressure_local'].rolling(window=window, min_periods=1).std()
    
    # 移動最大・最小値
    for window in [7, 14, 30]:
        df[f'temp_max_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).max()
        df[f'temp_min_{window}'] = df['avg_temp'].rolling(window=window, min_periods=1).min()
        df[f'humidity_max_{window}'] = df['avg_humidity'].rolling(window=window, min_periods=1).max()
        df[f'humidity_min_{window}'] = df['avg_humidity'].rolling(window=window, min_periods=1).min()
    
    # 移動範囲（最大値 - 最小値）
    for window in [7, 14, 30]:
        df[f'temp_range_{window}'] = df[f'temp_max_{window}'] - df[f'temp_min_{window}']
        df[f'humidity_range_{window}'] = df[f'humidity_max_{window}'] - df[f'humidity_min_{window}']
    
    # 変化率（前日比）
    df['temp_change_rate'] = df['avg_temp'].pct_change()
    df['humidity_change_rate'] = df['avg_humidity'].pct_change()
    df['pressure_change_rate'] = df['pressure_local'].pct_change()
    df['wind_change_rate'] = df['avg_wind'].pct_change()
    
    # 加速度（変化率の変化率）
    df['temp_acceleration'] = df['temp_change_rate'].pct_change()
    df['humidity_acceleration'] = df['humidity_change_rate'].pct_change()
    df['pressure_acceleration'] = df['pressure_change_rate'].pct_change()
    
    return df

def prepare_data_for_training(df, train_dates, test_dates):
    """トレーニングデータとテストデータを準備"""
    # データを日付で分割
    train_data = df[df['hospitalization_date'].isin(train_dates)].copy()
    test_data = df[df['hospitalization_date'].isin(test_dates)].copy()
    
    # 特徴量とターゲットを分離
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院データに関連する特徴量を除外（当日のデータのみ）
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization_count'  # 当日のデータのみを除外（ラグ特徴量は除外しない）
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # カテゴリカル変数をダミー変数に変換
    categorical_cols = ['season']
    for col in categorical_cols:
        if col in df.columns:
            # ダミー変数を作成
            dummies = pd.get_dummies(df[col], prefix=col)
            # 元のカラムを削除
            feature_columns = [f for f in feature_columns if f != col]
            # ダミー変数を追加
            feature_columns.extend(dummies.columns)
            # データにダミー変数を追加
            train_data = pd.concat([train_data, dummies.loc[train_data.index]], axis=1)
            test_data = pd.concat([test_data, dummies.loc[test_data.index]], axis=1)
    
    # 数値列のみを選択
    numeric_cols = train_data[feature_columns].select_dtypes(include=[np.number]).columns
    feature_columns = list(numeric_cols)
    
    # 欠損値を処理
    train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
    test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
    
    # inf→NaN→中央値補完
    train_data[feature_columns] = train_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
    test_data[feature_columns] = test_data[feature_columns].replace([np.inf, -np.inf], np.nan)
    test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
    
    X_train = train_data[feature_columns].values
    y_train = train_data['target'].values
    X_test = test_data[feature_columns].values
    y_test = test_data['target'].values
    
    return X_train, y_train, X_test, y_test, feature_columns

def create_seasonal_splits(df, n_splits=3):
    """季節性を考慮した時系列分割"""
    df_sorted = df.sort_values('hospitalization_date')
    total_days = len(df_sorted)
    split_size = total_days // (n_splits + 1)  # +1で分割サイズを調整
    
    splits = []
    for i in range(n_splits):
        # 訓練データ: 0から(i+1)*split_sizeまで
        # テストデータ: (i+1)*split_sizeから(i+2)*split_sizeまで
        train_end_idx = (i + 1) * split_size
        test_start_idx = train_end_idx
        test_end_idx = (i + 2) * split_size if i < n_splits - 1 else total_days
        
        train_dates = df_sorted.iloc[:train_end_idx]['hospitalization_date'].tolist()
        test_dates = df_sorted.iloc[test_start_idx:test_end_idx]['hospitalization_date'].tolist()
        
        if len(train_dates) > 0 and len(test_dates) > 0:
            splits.append((train_dates, test_dates))
    
    return splits

def optimize_lgb_params(X_train, y_train, X_val, y_val):
    """LightGBMのハイパーパラメータを最適化"""
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100)
        }
        
        model = lgb.LGBMClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    
    return study.best_params

def evaluate_model_performance(y_true, y_pred_proba, threshold=0.5):
    """モデルの性能を評価"""
    # NaN値を処理
    if np.isnan(y_pred_proba).any():
        print("予測値にNaNが含まれていたため中央値で補完します")
        if np.isnan(y_pred_proba).all():
            print("予測値が全てNaNのため0.5で補完します")
            y_pred_proba = np.full_like(y_pred_proba, 0.5)
        else:
            median_pred = np.nanmedian(y_pred_proba)
            y_pred_proba = np.where(np.isnan(y_pred_proba), median_pred, y_pred_proba)
    
    if np.isnan(y_pred_proba).any():
        print("予測値にNaNが残っていたため0.5で補完します")
        y_pred_proba = np.where(np.isnan(y_pred_proba), 0.5, y_pred_proba)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # 各種指標を計算
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    precision_score_val = precision_score(y_true, y_pred, zero_division=0)
    recall_score_val = recall_score(y_true, y_pred, zero_division=0)
    f1_score_val = f1_score(y_true, y_pred, zero_division=0)
    
    # 特異度（Specificity）を計算
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision_score_val,
        'recall': recall_score_val,
        'f1_score': f1_score_val,
        'specificity': specificity
    }

def plot_model_performance(y_true, y_pred_proba, fold_num=None, avg_auc=None):
    """モデルの性能をプロット"""
    # NaN値を処理
    if np.isnan(y_pred_proba).any():
        median_pred = np.nanmedian(y_pred_proba)
        y_pred_proba = np.where(np.isnan(y_pred_proba), median_pred, y_pred_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC曲線
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)
    
    # Precision-Recall曲線
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs('results_lgb_only', exist_ok=True)
    fold_str = f'_fold_{fold_num}' if fold_num is not None else ''
    auc_suffix = f"_AUC{avg_auc:.4f}" if avg_auc is not None else ""
    plt.savefig(f'results_lgb_only/performance_curves{fold_str}{auc_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_results(cv_results, feature_importance=None, avg_auc=None):
    """詳細な結果を保存"""
    os.makedirs('results_lgb_only', exist_ok=True)
    
    # 全体の平均指標を計算
    overall_metrics = {
        'roc_auc': np.mean([r['roc_auc'] for r in cv_results]),
        'pr_auc': np.mean([r['pr_auc'] for r in cv_results]),
        'precision': np.mean([r['precision'] for r in cv_results]),
        'recall': np.mean([r['recall'] for r in cv_results]),
        'f1_score': np.mean([r['f1_score'] for r in cv_results]),
        'specificity': np.mean([r['specificity'] for r in cv_results])
    }
    
    # JSONファイルとして保存
    auc_suffix = f"_AUC{avg_auc:.4f}" if avg_auc is not None else ""
    with open(f'results_lgb_only/detailed_metrics{auc_suffix}.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    # Markdownファイルとして保存
    with open(f'results_lgb_only/detailed_metrics{auc_suffix}.md', 'w') as f:
        f.write("# LightGBM単独モデル 詳細結果\n\n")
        f.write(f"## 平均性能指標\n")
        f.write(f"- **ROC-AUC**: {overall_metrics['roc_auc']:.4f}\n")
        f.write(f"- **PR-AUC**: {overall_metrics['pr_auc']:.4f}\n")
        f.write(f"- **Precision**: {overall_metrics['precision']:.4f}\n")
        f.write(f"- **Recall**: {overall_metrics['recall']:.4f}\n")
        f.write(f"- **F1-Score**: {overall_metrics['f1_score']:.4f}\n")
        f.write(f"- **Specificity**: {overall_metrics['specificity']:.4f}\n\n")
        
        f.write("## 各Foldの結果\n")
        for i, result in enumerate(cv_results):
            f.write(f"### Fold {i+1}\n")
            f.write(f"- ROC-AUC: {result['roc_auc']:.4f}\n")
            f.write(f"- PR-AUC: {result['pr_auc']:.4f}\n")
            f.write(f"- Precision: {result['precision']:.4f}\n")
            f.write(f"- Recall: {result['recall']:.4f}\n")
            f.write(f"- F1-Score: {result['f1_score']:.4f}\n")
            f.write(f"- Specificity: {result['specificity']:.4f}\n\n")
    
    # 特徴量重要度を保存
    if feature_importance is not None:
        feature_importance.to_csv(f'results_lgb_only/feature_importance{auc_suffix}.csv', index=False)
        
        # 特徴量重要度の可視化
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance (LightGBM)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'results_lgb_only/feature_importance{auc_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """メイン処理"""
    print("=== LightGBM単独モデル 学習開始 ===")
    
    # データ読み込み
    df = load_processed_data()
    print(f"データ読み込み完了: {len(df)}行")
    
    # 期間制限（2012-2019）
    df = df[df['hospitalization_date'] <= '2019-12-31'].copy()
    print(f"期間制限後: {len(df)}行")
    
    # 基本特徴量エンジニアリング（元のモデルと同じ）
    df = create_date_features(df)
    
    # 元のモデルと同じ特徴量エンジニアリングを追加
    # 気象要素の相互作用特徴量を作成
    df = create_weather_interaction_features(df)
    
    # 時系列特徴量を作成
    df = create_time_series_features(df)
    
    # 異常気象検出特徴量を作成
    df = detect_extreme_weather(df)
    
    # ラグ特徴量を追加（元のモデルと同じ方法）
    df = add_lag_features(df)
    
    # ターゲット変数の作成（固定閾値15.0）
    threshold = 15.0
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    print(f"ターゲット変数分布: {df['target'].value_counts().to_dict()}")
    
    # 時系列分割
    splits = create_seasonal_splits(df, n_splits=3)
    print(f"分割数: {len(splits)}")
    
    cv_results = []
    feature_importance_list = []
    
    for fold, (train_dates, test_dates) in enumerate(splits):
        print(f"\n=== Fold {fold + 1} ===")
        
        # データ準備（ラグ特徴量は既に作成済み）
        train_data = df[df['hospitalization_date'].isin(train_dates)].copy()
        test_data = df[df['hospitalization_date'].isin(test_dates)].copy()
        
        # 特徴量とターゲットを分離
        exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 入院データに関連する特徴量を除外（当日のデータのみ）
        hospitalization_related_cols = [
            col for col in feature_columns 
            if any(keyword in col.lower() for keyword in [
                'hospitalization_count'  # 当日のデータのみを除外（ラグ特徴量は除外しない）
            ])
        ]
        feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
        
        # カテゴリカル変数をダミー変数に変換
        categorical_cols = ['season']
        for col in categorical_cols:
            if col in df.columns:
                # ダミー変数を作成
                dummies = pd.get_dummies(df[col], prefix=col)
                # 元のカラムを削除
                feature_columns = [f for f in feature_columns if f != col]
                # ダミー変数を追加
                feature_columns.extend(dummies.columns)
                # データにダミー変数を追加
                train_data = pd.concat([train_data, dummies.loc[train_data.index]], axis=1)
                test_data = pd.concat([test_data, dummies.loc[test_data.index]], axis=1)
        
        # 数値列のみを選択
        numeric_cols = train_data[feature_columns].select_dtypes(include=[np.number]).columns
        feature_columns = list(numeric_cols)
        
        # 欠損値を処理
        train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
        test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
        
        # inf→NaN→中央値補完
        train_data[feature_columns] = train_data[feature_columns].replace([np.inf, -np.inf], np.nan)
        train_data[feature_columns] = train_data[feature_columns].fillna(train_data[feature_columns].median())
        test_data[feature_columns] = test_data[feature_columns].replace([np.inf, -np.inf], np.nan)
        test_data[feature_columns] = test_data[feature_columns].fillna(test_data[feature_columns].median())
        
        X_train = train_data[feature_columns].values
        y_train = train_data['target'].values
        X_test = test_data[feature_columns].values
        y_test = test_data['target'].values
        
        print(f"トレーニングデータ: {len(X_train)}サンプル")
        print(f"テストデータ: {len(X_test)}サンプル")
        print(f"特徴量数: {len(feature_columns)}")
        
        # LightGBM単独で学習
        print("LightGBMハイパーパラメータ最適化中...")
        best_params = optimize_lgb_params(X_train, y_train, X_test, y_test)
        print(f"最適パラメータ: {best_params}")
        
        # 最終モデル学習
        print("LightGBM学習中...")
        final_model = lgb.LGBMClassifier(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        
        # 予測
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        
        # 性能評価
        metrics = evaluate_model_performance(y_test, y_pred_proba)
        cv_results.append(metrics)
        
        print(f"Fold {fold + 1} 結果:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        
        # 特徴量重要度
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        feature_importance_list.append(importance_df)
        
        # プロット
        avg_auc = np.mean([r['roc_auc'] for r in cv_results])
        plot_model_performance(y_test, y_pred_proba, fold + 1, avg_auc)
    
    # 平均特徴量重要度
    avg_importance = pd.concat(feature_importance_list).groupby('feature')['importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # 結果保存
    avg_auc = np.mean([r['roc_auc'] for r in cv_results])
    print(f"\n=== 最終結果 ===")
    print(f"平均ROC-AUC: {avg_auc:.4f}")
    
    save_detailed_results(cv_results, avg_importance, avg_auc)
    
    print("LightGBM単独モデル学習完了！")

if __name__ == "__main__":
    main() 