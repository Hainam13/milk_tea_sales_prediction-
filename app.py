import streamlit as st
import pandas as pd
import pickle

# Load mô hình đã huấn luyện
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Hàm xử lý dữ liệu đầu vào
def preprocess_input(day_of_week, weather, is_promotion):
    # Mã hóa các giá trị đầu vào theo đúng định dạng model
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weathers = ['Sunny', 'Rainy', 'Cloudy']

    day_encoded = [1 if day == day_of_week else 0 for day in days]
    weather_encoded = [1 if w == weather else 0 for w in weathers]
    promo_encoded = [1 if is_promotion else 0]

    return pd.DataFrame([day_encoded + weather_encoded + promo_encoded])

# Giao diện Streamlit
st.title("📈 Dự đoán số lượng trà sữa bán ra")

day = st.selectbox("Chọn ngày trong tuần", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weather = st.selectbox("Chọn thời tiết", ['Sunny', 'Rainy', 'Cloudy'])
promotion = st.checkbox("Có khuyến mãi?")

if st.button("Dự đoán"):
    input_data = preprocess_input(day, weather, promotion)
    prediction = model.predict(input_data)
    st.success(f"🔮 Dự đoán: {int(prediction[0])} ly trà sữa")

