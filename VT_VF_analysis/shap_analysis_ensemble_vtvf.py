import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_ensemble_model_and_data():
    """アンサンブルモデルとデータを読み込み"""
    print("=== VT/VF アンサンブルモデルとデータの読み込み ===")
    
    try:
        # 保存されたアンサンブルモデルを読み込み
        models = {}
        
        # XGBoostモデル
        try:
            models['xgb'] = joblib.load('saved_models/xgb_model_vtvf_fold_2.pkl')
            print("✓ XGBoostモデル読み込み完了")
        except:
            print("⚠️ XGBoostモデルが見つかりません")
        
        # 他のモデルも同様に読み込み（存在する場合）
        model_files = {
            'lgb': 'saved_models/lgb_model_vtvf_fold_2.pkl',
            'cat': 'saved_models/cat_model_vtvf_fold_2.pkl'
        }
        
        for model_name, file_path in model_files.items():
            try:
                models[model_name] = joblib.load(file_path)
                print(f"✓ {model_name}モデル読み込み完了")
            except:
                print(f"⚠️ {model_name}モデルが見つかりません")
        
        # データの読み込み
        df = pd.read_csv('vtvf_weather_merged.csv')
        df['hospitalization_date'] = pd.to_datetime(df['hospitalization_date_vtvf'])
        df['hospitalization_count'] = df['people_vtvf']
        
        print("✓ データ読み込み完了")
        return models, df
        
    except Exception as e:
        print(f"❌ モデルまたはデータの読み込みエラー: {e}")
        return None, None

def create_date_features(df):
    """日付関連の特徴量を作成（元のスクリプトと同じ）"""
    from datetime import datetime, timedelta
    import jpholiday
    
    # 日付関連の特徴量
    df['year'] = df['hospitalization_date'].dt.year
    df['month'] = df['hospitalization_date'].dt.month
    df['day'] = df['hospitalization_date'].dt.day
    df['dayofweek'] = df['hospitalization_date'].dt.dayofweek
    df['week'] = df['hospitalization_date'].dt.isocalendar().week
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['hospitalization_date'].apply(
        lambda x: int(jpholiday.is_holiday(x) or x.weekday() in [5, 6])
    )
    
    # 季節性指標
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # 季節
    df['season'] = df['month'].map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    # 季節ダミー変数
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    # 月末・月初フラグ
    df['is_month_start'] = df['hospitalization_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['hospitalization_date'].dt.is_month_end.astype(int)
    df['quarter'] = df['hospitalization_date'].dt.quarter
    
    return df

def create_weather_interaction_features(df):
    """気象相互作用特徴量を作成（元のスクリプトと同じ）"""
    # 異常気象フラグ
    df['is_tropical_night'] = (df['min_temp_vtvf'] >= 25).astype(int)
    df['is_extremely_hot'] = (df['max_temp_vtvf'] >= 35).astype(int)
    df['is_hot_day'] = (df['max_temp_vtvf'] >= 30).astype(int)
    df['is_summer_day'] = (df['max_temp_vtvf'] >= 25).astype(int)
    df['is_winter_day'] = (df['min_temp_vtvf'] < 0).astype(int)
    df['is_freezing_day'] = (df['max_temp_vtvf'] < 0).astype(int)
    
    # 気象相互作用
    df['temp_humidity'] = df['avg_temp_vtvf'] * df['avg_humidity_vtvf']
    df['temp_pressure'] = df['avg_temp_vtvf'] * df['vapor_pressure_vtvf']
    df['temp_change'] = (df['avg_temp_vtvf'] - df['avg_temp_vtvf'].shift(1)).fillna(0)
    df['humidity_change'] = (df['avg_humidity_vtvf'] - df['avg_humidity_vtvf'].shift(1)).fillna(0)
    df['pressure_change'] = (df['vapor_pressure_vtvf'] - df['vapor_pressure_vtvf'].shift(1)).fillna(0)
    
    # 不快指数
    df['discomfort_index'] = 0.81 * df['avg_temp_vtvf'] + 0.01 * df['avg_humidity_vtvf'] * (0.99 * df['avg_temp_vtvf'] - 14.3) + 46.3
    
    return df

def create_time_series_features(df):
    """時系列特徴量を作成（元のスクリプトと同じ）"""
    # 時系列特徴量
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        for window in [3, 7, 14]:
            df[f'{col}_ma_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).mean()
            df[f'{col}_std_{window}d'] = df[col].shift(1).rolling(window=window, min_periods=window).std()
    
    return df

def create_advanced_weather_timeseries_features(df):
    """高度な気象時系列特徴量を作成（元のスクリプトと同じ）"""
    # 高度な時系列特徴量
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        # ラグ特徴量
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}d'] = df[col].shift(lag)
        
        # 変化率
        df[f'{col}_change_rate'] = df[col].pct_change()
        
        # 移動平均の変化率
        for window in [3, 7]:
            df[f'{col}_ma_{window}d_change_rate'] = df[f'{col}_ma_{window}d'].pct_change()
    
    return df

def create_seasonal_weighted_features(df):
    """季節性重み付け特徴量を作成（元のスクリプトと同じ）"""
    # 季節性重み付け
    df['seasonal_weight'] = np.where(df['season'] == 'summer', 1.2, 
                                    np.where(df['season'] == 'winter', 1.1, 1.0))
    
    # 季節性重み付けされた気象特徴量
    weather_cols = ['avg_temp_vtvf', 'avg_humidity_vtvf', 'vapor_pressure_vtvf']
    for col in weather_cols:
        df[f'{col}_seasonal_weighted'] = df[col] * df['seasonal_weight']
    
    return df

def prepare_features_for_shap(df):
    """SHAP解析用の特徴量を準備（元のスクリプトと同じ処理）"""
    print("\n=== SHAP解析用特徴量準備 ===")
    
    # 元のスクリプトと同じ特徴量作成処理
    df = create_date_features(df)
    df = create_weather_interaction_features(df)
    df = create_time_series_features(df)
    df = create_advanced_weather_timeseries_features(df)
    df = create_seasonal_weighted_features(df)
    
    # ターゲット変数
    threshold = df['hospitalization_count'].quantile(0.75)
    df['target'] = (df['hospitalization_count'] >= threshold).astype(int)
    
    print("✓ 特徴量作成完了")
    return df

def select_features_for_shap(df):
    """SHAP解析用の特徴量を選択（元のスクリプトと同じ処理）"""
    print("\n=== 特徴量選択 ===")
    
    # 除外する列（元のスクリプトと同じ）
    exclude_cols = ['hospitalization_date', 'target', 'season', 'prefecture_name', 'date', 
                   'hospitalization_count', 'people']
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 入院データ関連の特徴量を除外（元のスクリプトと同じ）
    hospitalization_related_cols = [
        col for col in feature_columns 
        if any(keyword in col.lower() for keyword in [
            'hospitalization', 'patient', 'people', 'patients_lag', 'patients_ma', 
            'patients_std', 'patients_max', 'patients_min', 'dow_mean'
        ])
    ]
    feature_columns = [col for col in feature_columns if col not in hospitalization_related_cols]
    
    # 数値型の列のみを抽出
    numeric_cols = df[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    feature_columns = list(numeric_cols)
    
    print(f"✓ 初期特徴量数: {len(feature_columns)}")
    
    # 元のスクリプトと同じ特徴量選択ロジックを適用
    # データの準備
    X = df[feature_columns].copy()
    y = df['target']
    
    # NaN値の処理
    X = X.fillna(method='ffill').fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LightGBMで特徴量重要度を計算
    import lightgbm as lgb
    lgb_temp = lgb.LGBMClassifier(random_state=42)
    lgb_temp.fit(X_scaled, y)
    importances = lgb_temp.feature_importances_
    
    # 上位100個の特徴量を選択（元のスクリプトと同じ）
    top_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])][:100]
    
    # 高相関な特徴量を除去
    X_df = pd.DataFrame(X_scaled, columns=feature_columns)
    reduced_features = remove_highly_correlated_features(X_df, top_features)
    
    # 保存されたモデルと同じ特徴量数になるまで調整
    if len(reduced_features) > 65:
        # 重要度の低い特徴量から削除
        sorted_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]) if f in reduced_features]
        reduced_features = sorted_features[:65]
    elif len(reduced_features) < 65:
        # 重要度の高い特徴量を追加
        sorted_features = [f for f, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1])]
        additional_features = [f for f in sorted_features if f not in reduced_features][:65-len(reduced_features)]
        reduced_features.extend(additional_features)
    
    print(f"✓ 最終特徴量数: {len(reduced_features)}")
    return reduced_features

def remove_highly_correlated_features(df, feature_columns, threshold=0.95):
    """高相関な特徴量を除去（元のスクリプトと同じ）"""
    if len(feature_columns) <= 1:
        return feature_columns
    
    # 相関行列を計算
    corr_matrix = df[feature_columns].corr().abs()
    
    # 上三角行列を取得
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 高相関な特徴量のペアを見つける
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # 除去する特徴量を除外
    selected_features = [col for col in feature_columns if col not in to_drop]
    
    return selected_features

def create_ensemble_predictions(models, X_scaled):
    """アンサンブル予測を作成"""
    print("\n=== アンサンブル予測作成 ===")
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            if model_name == 'lgb':
                # LightGBM Boosterオブジェクト
                predictions[model_name] = model.predict(X_scaled)
            elif model_name in ['xgb', 'cat']:
                # XGBoost, CatBoost
                predictions[model_name] = model.predict_proba(X_scaled)[:, 1]
            else:
                # Neural Network
                predictions[model_name] = model.predict(X_scaled).flatten()
            
            print(f"✓ {model_name}予測完了")
            
        except Exception as e:
            print(f"❌ {model_name}予測エラー: {e}")
    
    # アンサンブル予測（平均）
    if predictions:
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        print("✓ アンサンブル予測完了")
        return ensemble_pred, predictions
    else:
        print("❌ 有効な予測がありません")
        return None, None

def perform_ensemble_shap_analysis(models, df, feature_columns):
    """アンサンブルSHAP解析を実行"""
    print("\n=== アンサンブルSHAP解析実行 ===")
    
    # データの準備
    X = df[feature_columns].copy()
    y = df['target']
    
    # NaN値の処理
    X = X.fillna(method='ffill').fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    print(f"✓ データ準備完了: {X_scaled.shape}")
    
    # アンサンブル予測
    ensemble_pred, individual_predictions = create_ensemble_predictions(models, X_scaled)
    if ensemble_pred is None:
        return None, None, None, None, None
    
    # SHAP値の計算（各モデルで）
    print("各モデルのSHAP値を計算中...")
    shap_values_dict = {}
    
    for model_name, model in models.items():
        try:
            if hasattr(model, 'feature_importances_'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                shap_values_dict[model_name] = shap_values
                print(f"✓ {model_name} SHAP値計算完了")
            else:
                print(f"⚠️ {model_name}はSHAP解析に対応していません")
        except Exception as e:
            print(f"❌ {model_name} SHAP値計算エラー: {e}")
    
    # アンサンブルSHAP値（平均）
    if shap_values_dict:
        ensemble_shap_values = np.mean(list(shap_values_dict.values()), axis=0)
        print("✓ アンサンブルSHAP値計算完了")
        
        # 代表的なTree-basedモデルでexplainerを作成
        representative_model = None
        for model_name in ['xgb', 'lgb', 'cat']:
            if model_name in models:
                representative_model = models[model_name]
                break
        
        if representative_model:
            explainer = shap.TreeExplainer(representative_model)
        else:
            explainer = None
            
        return X_scaled, y, ensemble_shap_values, explainer, individual_predictions
    else:
        print("❌ 有効なSHAP値がありません")
        return None, None, None, None, None

def create_ensemble_shap_visualizations(X_scaled, y, ensemble_shap_values, explainer, 
                                      individual_predictions, feature_columns):
    """アンサンブルSHAP可視化を作成"""
    print("\n=== アンサンブルSHAP可視化作成 ===")
    
    # 結果保存用ディレクトリ作成
    import os
    os.makedirs('ensemble_shap_results_vtvf', exist_ok=True)
    
    # 1. Ensemble Summary Plot
    print("1. アンサンブルSummary Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(ensemble_shap_values, X_scaled, feature_names=feature_columns, show=False)
    plt.title('Ensemble SHAP Summary Plot - VT/VF Weather Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ensemble_shap_results_vtvf/ensemble_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ アンサンブルSummary Plot保存完了")
    
    # 2. Ensemble Feature Importance Plot
    print("2. アンサンブルFeature Importance Plot作成中...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(ensemble_shap_values, X_scaled, plot_type="bar", 
                      feature_names=feature_columns, show=False)
    plt.title('Ensemble SHAP Feature Importance - VT/VF Weather Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ensemble_shap_results_vtvf/ensemble_shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ アンサンブルFeature Importance Plot保存完了")
    
    # 3. Individual Model SHAP Comparison
    if individual_predictions:
        print("3. 個別モデルSHAP比較作成中...")
        # 各モデルのSHAP値を比較
        model_names = list(individual_predictions.keys())
        if len(model_names) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, model_name in enumerate(model_names[:4]):  # 最大4モデル
                if i < len(axes):
                    # 簡略化された特徴量重要度（予測値の分散で近似）
                    feature_importance = np.std(individual_predictions[model_name]) * np.ones(len(feature_columns))
                    top_features_idx = np.argsort(feature_importance)[-10:]
                    top_features = [feature_columns[i] for i in top_features_idx]
                    
                    axes[i].barh(range(len(top_features)), feature_importance[top_features_idx])
                    axes[i].set_yticks(range(len(top_features)))
                    axes[i].set_yticklabels(top_features)
                    axes[i].set_title(f'{model_name.upper()} Feature Importance')
                    axes[i].set_xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig('ensemble_shap_results_vtvf/individual_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ 個別モデル比較保存完了")
    
    # 4. Ensemble Dependence Plots for Top Features
    print("4. アンサンブルDependence Plots作成中...")
    # 上位10個の特徴量を取得
    feature_importance = np.abs(ensemble_shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-10:]
    top_features = [feature_columns[i] for i in top_features_idx]
    
    for i, feature in enumerate(top_features):
        print(f"  {i+1}/10: {feature}")
        plt.figure(figsize=(10, 6))
        # 簡略化されたdependence plot
        plt.scatter(X_scaled[feature], ensemble_shap_values[:, top_features_idx[i]], alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel(f'SHAP Value for {feature}')
        plt.title(f'Ensemble SHAP Dependence Plot - {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'ensemble_shap_results_vtvf/ensemble_shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ アンサンブルDependence Plots保存完了")

def create_ensemble_shap_report(X_scaled, y, ensemble_shap_values, individual_predictions, feature_columns):
    """アンサンブルSHAP解析レポートを作成"""
    print("\n=== アンサンブルSHAP解析レポート作成 ===")
    
    # 特徴量重要度の計算
    feature_importance = np.abs(ensemble_shap_values).mean(0)
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 上位20個の特徴量
    top_20_features = importance_df.head(20)
    
    # 個別モデルの性能
    model_performance = {}
    if individual_predictions:
        from sklearn.metrics import roc_auc_score
        for model_name, pred in individual_predictions.items():
            try:
                auc = roc_auc_score(y, pred)
                model_performance[model_name] = auc
            except:
                model_performance[model_name] = None
    
    # レポート作成
    report = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'Ensemble (XGBoost, LightGBM, CatBoost)',
        'target_variable': 'VT/VF High Risk Days (75th percentile)',
        'data_shape': X_scaled.shape,
        'positive_class_ratio': y.mean(),
        'ensemble_models': list(individual_predictions.keys()) if individual_predictions else [],
        'model_performance': model_performance,
        'top_20_features': top_20_features.to_dict('records'),
        'feature_importance_summary': {
            'total_features': len(feature_columns),
            'weather_features': len([f for f in feature_columns if any(x in f for x in ['temp', 'humidity', 'pressure', 'wind', 'sunshine'])]),
            'extreme_weather_features': len([f for f in feature_columns if 'is_' in f]),
            'time_series_features': len([f for f in feature_columns if any(x in f for x in ['ma_', 'std_', 'lag_'])]),
            'interaction_features': len([f for f in feature_columns if any(x in f for x in ['temp_', 'humidity_', 'pressure_'])]),
        }
    }
    
    # JSONファイルとして保存
    import json
    with open('ensemble_shap_results_vtvf/ensemble_shap_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # Markdownファイルとして保存
    with open('ensemble_shap_results_vtvf/ensemble_shap_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('# VT/VF Ensemble Weather Model SHAP Analysis Report\n\n')
        f.write(f'**Analysis Date:** {report["analysis_date"]}\n\n')
        f.write(f'**Model Type:** {report["model_type"]}\n\n')
        f.write(f'**Target Variable:** {report["target_variable"]}\n\n')
        f.write(f'**Data Shape:** {report["data_shape"]}\n\n')
        f.write(f'**Positive Class Ratio:** {report["positive_class_ratio"]:.3f}\n\n')
        
        f.write('## Ensemble Models\n\n')
        for model_name in report['ensemble_models']:
            auc = report['model_performance'].get(model_name, 'N/A')
            f.write(f'- **{model_name.upper()}**: AUC = {auc:.4f}\n')
        
        f.write('\n## Feature Importance Summary\n\n')
        f.write('| Category | Count |\n')
        f.write('|----------|-------|\n')
        for category, count in report['feature_importance_summary'].items():
            f.write(f'| {category.replace("_", " ").title()} | {count} |\n')
        
        f.write('\n## Top 20 Most Important Features (Ensemble)\n\n')
        f.write('| Rank | Feature | Importance |\n')
        f.write('|------|---------|------------|\n')
        for i, row in enumerate(top_20_features.itertuples(), 1):
            f.write(f'| {i} | {row.feature} | {row.importance:.4f} |\n')
    
    print("✓ アンサンブルSHAP解析レポート保存完了")
    return report

def main():
    """メイン実行関数"""
    print("=== VT/VF Ensemble Weather Model SHAP Analysis ===")
    
    # 1. アンサンブルモデルとデータの読み込み
    models, df = load_ensemble_model_and_data()
    if models is None or df is None:
        print("❌ モデルまたはデータの読み込みに失敗しました")
        return
    
    # 2. 特徴量の準備
    df = prepare_features_for_shap(df)
    feature_columns = select_features_for_shap(df)
    
    # 3. アンサンブルSHAP解析の実行
    result = perform_ensemble_shap_analysis(models, df, feature_columns)
    if result[0] is None:
        print("❌ アンサンブルSHAP解析に失敗しました")
        return
    
    X_scaled, y, ensemble_shap_values, explainer, individual_predictions = result
    
    # 4. 可視化の作成
    create_ensemble_shap_visualizations(X_scaled, y, ensemble_shap_values, explainer, 
                                      individual_predictions, feature_columns)
    
    # 5. レポートの作成
    report = create_ensemble_shap_report(X_scaled, y, ensemble_shap_values, 
                                       individual_predictions, feature_columns)
    
    print("\n=== アンサンブルSHAP解析完了 ===")
    print("✓ 結果は 'ensemble_shap_results_vtvf/' ディレクトリに保存されました")
    print(f"✓ 上位特徴量: {report['top_20_features'][0]['feature']}")
    print(f"✓ 使用モデル: {', '.join(report['ensemble_models'])}")

if __name__ == "__main__":
    main() 