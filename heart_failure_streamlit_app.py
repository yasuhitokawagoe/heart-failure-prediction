import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime
import numpy as np

# ページ設定
st.set_page_config(
    page_title="心不全リスク予測システム",
    page_icon="💓",
    layout="wide"
)

# モデル読み込み（フォールバック用の簡易モデル）
def load_model():
    try:
        # 実際のモデルファイルがある場合は読み込み
        model = joblib.load("heart_failure_model.pkl")
        st.success("✓ 心不全予測モデルを読み込みました")
        return model
    except FileNotFoundError:
        st.warning("⚠️ モデルファイルが見つかりません。簡易版モデルを使用します。")
        # 簡易版モデル（実際のモデルがない場合の代替）
        return None

# Open-Meteo APIから天気データを取得する関数
def get_weather_forecast():
    try:
        lat, lon = 35.6895, 139.6917  # 東京の緯度経度
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min,surface_pressure_max,surface_pressure_min&timezone=Asia%2FTokyo"
        
        with st.spinner("Open-Meteo APIから気象データを取得中..."):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

        # 今日のデータを抽出
        weather_info = {
            "temperature_2m_max": data["daily"]["temperature_2m_max"][0],
            "temperature_2m_min": data["daily"]["temperature_2m_min"][0],
            "relative_humidity_2m_max": data["daily"]["relative_humidity_2m_max"][0],
            "relative_humidity_2m_min": data["daily"]["relative_humidity_2m_min"][0],
            "surface_pressure_max": data["daily"]["surface_pressure_max"][0],
            "surface_pressure_min": data["daily"]["surface_pressure_min"][0],
        }

        st.success("✓ Open-Meteo APIから実際の気象データを取得しました")
        return weather_info
        
    except Exception as e:
        st.error(f"気象データ取得エラー: {e}")
        # フォールバック用の推定データ
        return {
            "temperature_2m_max": 28.5,
            "temperature_2m_min": 22.0,
            "relative_humidity_2m_max": 75,
            "relative_humidity_2m_min": 65,
            "surface_pressure_max": 1013,
            "surface_pressure_min": 1008,
        }

# モデルが期待する形式に整形
def format_weather_data(weather_info):
    weather_df = pd.DataFrame({
        "気温_max": [weather_info["temperature_2m_max"]],
        "気温_min": [weather_info["temperature_2m_min"]],
        "湿度_max": [weather_info["relative_humidity_2m_max"]],
        "湿度_min": [weather_info["relative_humidity_2m_min"]],
        "気圧_max": [weather_info["surface_pressure_max"]],
        "気圧_min": [weather_info["surface_pressure_min"]],
    })
    return weather_df

# 簡易版リスク予測関数（実際のモデルがない場合）
def predict_heart_failure_risk_simple(weather_df):
    """簡易版の心不全リスク予測ロジック"""
    temp_max = weather_df["気温_max"].iloc[0]
    temp_min = weather_df["気温_min"].iloc[0]
    humidity_max = weather_df["湿度_max"].iloc[0]
    pressure_max = weather_df["気圧_max"].iloc[0]
    
    risk_score = 0.0
    
    # 暑熱ストレス
    if temp_max > 30:
        risk_score += 0.3
    elif temp_max > 25:
        risk_score += 0.15
    
    # 寒冷ストレス
    if temp_min < 5:
        risk_score += 0.4
    elif temp_min < 10:
        risk_score += 0.2
    
    # 湿度ストレス
    if humidity_max > 80:
        risk_score += 0.1
    
    # 気圧変化
    if pressure_max < 1000:
        risk_score += 0.1
    
    # 温度変化
    temp_range = temp_max - temp_min
    if temp_range > 15:
        risk_score += 0.2
    elif temp_range > 10:
        risk_score += 0.1
    
    return min(risk_score, 1.0)

# リスクレベル判定
def get_risk_level(risk_score):
    if risk_score < 0.3:
        return "低リスク", "🟢"
    elif risk_score < 0.7:
        return "中リスク", "🟡"
    else:
        return "高リスク", "🔴"

# 推奨事項生成
def generate_recommendations(weather_df, risk_level):
    recommendations = []
    
    temp_max = weather_df["気温_max"].iloc[0]
    temp_min = weather_df["気温_min"].iloc[0]
    humidity_max = weather_df["湿度_max"].iloc[0]
    
    if risk_level == "高リスク":
        recommendations.append("⚠️ 心不全患者の方は特に注意が必要です")
        recommendations.append("🏥 医療機関への連絡を検討してください")
        recommendations.append("😴 安静を保ち、過度な運動を避けてください")
    
    if temp_max > 30:
        recommendations.append("🌡️ 暑熱ストレスが検出されました。適切な水分補給と冷房を使用してください")
    
    if temp_min < 5:
        recommendations.append("❄️ 寒冷ストレスが検出されました。暖房を適切に使用してください")
    
    if humidity_max > 80:
        recommendations.append("💧 高湿度環境です。除湿機の使用を検討してください")
    
    if temp_max - temp_min > 15:
        recommendations.append("🌡️ 急激な温度変化が予想されます。体調管理に注意してください")
    
    if not recommendations:
        recommendations.append("✅ 現在の気象条件は心不全リスクが低い状態です")
    
    return recommendations

# Streamlit アプリの表示
def main():
    st.title("💓 心不全リスク予測システム")
    st.markdown("### Open-Meteo API から取得した東京の気象データを使用")
    
    # サイドバー
    st.sidebar.header("設定")
    st.sidebar.markdown("**データソース**: Open-Meteo API")
    st.sidebar.markdown("**対象地域**: 東京")
    st.sidebar.markdown("**更新時刻**: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # メインコンテンツ
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 気象データ取得状況")
        
        # 天気データの取得と表示
        weather_info = get_weather_forecast()
        weather_df = format_weather_data(weather_info)
        
        # 気象データの表示
        st.subheader("🌤️ 今日の気象データ（APIから取得）")
        
        # データを美しく表示
        col_temp, col_humidity, col_pressure = st.columns(3)
        
        with col_temp:
            st.metric("最高気温", f"{weather_info['temperature_2m_max']:.1f}°C")
            st.metric("最低気温", f"{weather_info['temperature_2m_min']:.1f}°C")
        
        with col_humidity:
            st.metric("最高湿度", f"{weather_info['relative_humidity_2m_max']:.0f}%")
            st.metric("最低湿度", f"{weather_info['relative_humidity_2m_min']:.0f}%")
        
        with col_pressure:
            st.metric("最高気圧", f"{weather_info['surface_pressure_max']:.0f}hPa")
            st.metric("最低気圧", f"{weather_info['surface_pressure_min']:.0f}hPa")
        
        # 詳細データテーブル
        with st.expander("📋 詳細データ"):
            st.dataframe(weather_df)
    
    with col2:
        st.subheader("🎯 リスク予測結果")
        
        # モデル読み込み
        model = load_model()
        
        # リスク予測
        if model:
            try:
                risk_score = model.predict_proba(weather_df)[0][1]
            except:
                risk_score = predict_heart_failure_risk_simple(weather_df)
        else:
            risk_score = predict_heart_failure_risk_simple(weather_df)
        
        risk_level, risk_icon = get_risk_level(risk_score)
        
        # リスク表示
        st.markdown(f"### {risk_icon} {risk_level}")
        st.metric("リスク確率", f"{risk_score:.1%}")
        
        # 推奨事項
        st.subheader("💡 推奨事項")
        recommendations = generate_recommendations(weather_df, risk_level)
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # フッター
    st.markdown("---")
    st.markdown("**データソース**: [Open-Meteo API](https://open-meteo.com/)")
    st.markdown("**更新**: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main() 