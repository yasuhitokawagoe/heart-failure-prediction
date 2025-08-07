import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
import joblib
import matplotlib.pyplot as plt
import warnings
import optuna
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

def load_processed_data(file_path='../東京AMI天候入院人数込みモデル天気詳細追加後/東京AMI天気データとJROAD結合後2012年4月1日から2021年12月31日天気概況整理.csv'):
    """処理済みデータの読み込み"""
    from data_preprocessing import load_data, preprocess_data
    
    # データ読み込みと前処理
    df = load_data(file_path)
    processed_df, feature_columns = preprocess_data(df)
    
    return processed_df, feature_columns

def create_time_based_splits(df, n_splits=5):
    """時系列に基づく訓練・検証データの分割"""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(df) * 0.2))
    return tscv.split(df)

def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoostモデルの学習"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 100
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBMモデルの学習"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 3,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 100
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def train_random_forest(X_train, y_train):
    """ランダムフォレストモデルの学習"""
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'min_samples_split': 5,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """モデルの評価"""
    # 予測確率の取得
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 評価指標の計算
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Performance:")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score: {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return roc_auc, pr_auc

def plot_feature_importance(model, feature_names, model_name):
    """特徴量の重要度をプロット"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):
        importances = model.get_score(importance_type='gain')
    else:
        return
    
    # 重要度の降順でソート
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances ({model_name})')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'../results/{model_name.lower()}_feature_importance.png')
    plt.close()

def main():
    """メイン処理"""
    print("データの準備を開始...")
    df, feature_columns = load_processed_data()
    
    # 特徴量とターゲットの準備
    X = df[feature_columns]
    y = df['target']
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"y unique values: {y.unique()}")
    print(f"Feature columns count: {len(feature_columns)}")
    
    # データの欠損値チェック
    print(f"X missing values: {X.isnull().sum().sum()}")
    print(f"y missing values: {y.isnull().sum()}")
    
    # 時系列分割の作成
    splits = create_time_based_splits(df)
    
    # モデルの評価結果を保存
    results = []
    
    # 各分割でモデルを学習・評価
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\nFold {fold}")
        
        # データの分割
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_train unique: {y_train.unique()}")
        
        # 検証データの作成（訓練データの最後の20%）
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"After split - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"After split - X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        # 各モデルの学習と評価
        models = {
            'XGBoost': train_xgboost(X_train, y_train, X_val, y_val),
            'LightGBM': train_lightgbm(X_train, y_train, X_val, y_val),
            'RandomForest': train_random_forest(X_train, y_train)
        }
        
        # 各モデルの評価
        fold_results = {}
        for name, model in models.items():
            roc_auc, pr_auc = evaluate_model(model, X_test, y_test, name)
            fold_results[name] = {'roc_auc': roc_auc, 'pr_auc': pr_auc}
            
            # 特徴量の重要度をプロット
            plot_feature_importance(model, feature_columns, f"{name}_fold{fold}")
        
        results.append(fold_results)
    
    # 全フォールドの平均性能を計算
    print("\nAverage Performance Across Folds:")
    for model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
        avg_roc_auc = np.mean([fold[model_name]['roc_auc'] for fold in results])
        avg_pr_auc = np.mean([fold[model_name]['pr_auc'] for fold in results])
        print(f"\n{model_name}:")
        print(f"Average ROC-AUC: {avg_roc_auc:.4f}")
        print(f"Average PR-AUC: {avg_pr_auc:.4f}")
    
    # 最終モデルの保存（全データで学習）
    final_models = {
        'XGBoost': train_xgboost(X, y, X_val, y_val),
        'LightGBM': train_lightgbm(X, y, X_val, y_val),
        'RandomForest': train_random_forest(X, y)
    }
    
    # 最良モデルの選択と保存
    best_model_name = None
    best_auc = 0
    
    for name, model in final_models.items():
        # 全データでの評価
        y_pred_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
        
        print(f"{name} Final AUC: {auc:.4f}")
    
    # 最良モデルを保存
    best_model = final_models[best_model_name]
    joblib.dump(best_model, f'auc0.83_weather_flags_model_{best_auc:.4f}.pkl')
    
    print(f"\n最良モデル: {best_model_name} (AUC: {best_auc:.4f})")
    print("モデルの学習と評価が完了しました。")
    
    return best_model, best_auc

if __name__ == "__main__":
    best_model, best_auc = main()
    print(f"\n🏆 最終結果: AUC {best_auc:.4f}") 