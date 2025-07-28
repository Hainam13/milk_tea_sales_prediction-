# train_model.py

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Bước 1: Tạo dữ liệu đầu vào giả định (cần đúng format với app.py)
X = pd.DataFrame({
    'Monday':    [1, 0, 0],
    'Tuesday':   [0, 1, 0],
    'Wednesday': [0, 0, 1],
    'Thursday':  [0, 0, 0],
    'Friday':    [0, 0, 0],
    'Saturday':  [0, 0, 0],
    'Sunday':    [0, 0, 0],
    'Sunny':     [1, 0, 0],
    'Rainy':     [0, 1, 0],
    'Cloudy':    [0, 0, 1],
    'Promotion': [1, 0, 1]
})

# Bước 2: Dữ liệu đầu ra tương ứng (số lượng trà sữa bán)
y = [180, 90, 130]

# Bước 3: Huấn luyện mô hình
model = RandomForestRegressor()
model.fit(X, y)

# Bước 4: Lưu mô hình thành file model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model đã được lưu thành công vào 'model.pkl'")
