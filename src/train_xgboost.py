import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys
import numpy as np
from xgboost import XGBRegressor
import os

def train_xgboost_model(product_id=None):
    # Đọc dữ liệu đã tiền xử lý
    df = pd.read_csv("../data/data/processed_sales.csv")

    # Lọc theo product_id nếu có
    if product_id is not None:
        df = df[df['product_id'] == product_id]
        if df.empty:
            print(f"Không tìm thấy dữ liệu cho product_id={product_id}")
            return None

    # 👉 Gộp dữ liệu theo tháng
    df_monthly = df.groupby(['year', 'month', 'product_id']).agg({
        'day': 'mean',
        'dayofweek': 'mean',
        'price': 'mean',
        'discount_price': 'mean',
        'stock': 'mean',
        'sold': 'mean',
        'quantity_sold': 'sum',           # Tổng số sản phẩm đã bán trong tháng
        'category_code': 'first',
        'trend': 'mean',
        'is_holiday': 'sum'               # Tổng số ngày nghỉ trong tháng
    }).reset_index()

    # ✅ Danh sách đặc trưng và mục tiêu
    features = [
        'year', 'month', 'day', 'dayofweek',
        'price', 'discount_price', 'stock', 'sold',
        'category_code', 'trend', 'is_holiday'
    ]
    target = 'quantity_sold'

    # Kiểm tra thiếu giá trị
    df_monthly = df_monthly.dropna(subset=features + [target])

    X = df_monthly[features]
    y = df_monthly[target]

    if len(X) < 2:
        print(f"Dữ liệu sau khi gộp không đủ để train (cần >= 2 mẫu).")
        return None

    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Số mẫu train: {len(X_train)}")
    print(f"Số mẫu test: {len(X_test)}")

    # Huấn luyện mô hình
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    tolerance = 0.1
    correct_preds = sum(abs(y_pred - y_test) <= tolerance * y_test)
    accuracy = correct_preds / len(y_test)

    print(f"R^2 trên tập test: {score:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy (sai số <=10%): {accuracy:.4f}")

    print("\nSo sánh dự đoán và giá trị thực tế (5 dòng đầu):")
    for i in range(min(5, len(y_test))):
        print(f"Dự đoán: {y_pred[i]:.2f}  |  Thực tế: {y_test.values[i]}")

    # Lưu model
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_xgboost_sales.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel đã được lưu tại: {model_path}")

    return {
        "model": "XGBoost",
        "mae": mae,
        "rmse": rmse,
        "r2": score,
        "accuracy": accuracy
    }

if __name__ == "__main__":
    pid = None
    if len(sys.argv) > 1:
        try:
            pid = int(sys.argv[1])
        except ValueError:
            print("product_id phải là số nguyên.")
            sys.exit(1)

    result = train_xgboost_model(pid)
    if result is not None:
        print("\nĐánh giá model:")
        for k, v in result.items():
            print(f"{k}: {v}")
