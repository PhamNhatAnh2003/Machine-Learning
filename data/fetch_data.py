import requests
import pandas as pd
import os

# Gọi API từ Laravel
response = requests.get("http://127.0.0.1:8000/api/sales-data")

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)

# Lấy dữ liệu JSON
data = response.json()

# Đưa vào DataFrame và lưu
df = pd.DataFrame(data)
df.to_csv("data/sales.csv", index=False)

print("Dữ liệu đã được lưu vào data/sales.csv")
