import pandas as pd
import joblib
import sys
import os

def predict_monthly_sales(model, df, product_id):
    df_product = df[df['product_id'] == product_id]
    monthly_features = df_product.groupby(['year', 'month']).agg({
        'day': 'mean',
        'dayofweek': 'mean',
        'price': 'mean',
        'discount_price': 'mean',
        'stock': 'mean',
        'sold': 'mean',
        'category_code': 'first',
    }).reset_index()

    features = ['year', 'month', 'day', 'dayofweek', 'price', 'discount_price', 'stock', 'sold', 'category_code']
    X_monthly = monthly_features[features]

    y_pred = model.predict(X_monthly)

    return monthly_features, y_pred

def predict_next_month(model, monthly_features):
    last_row = monthly_features.iloc[-1]
    year, month = int(last_row['year']), int(last_row['month'])

    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    next_month_data = {
        'year': next_year,
        'month': next_month,
        'day': last_row['day'],
        'dayofweek': last_row['dayofweek'],
        'price': last_row['price'],
        'discount_price': last_row['discount_price'],
        'stock': last_row['stock'],
        'sold': last_row['sold'],
        'category_code': last_row['category_code'],
    }

    next_month_df = pd.DataFrame([next_month_data])
    features = ['year', 'month', 'day', 'dayofweek', 'price', 'discount_price', 'stock', 'sold', 'category_code']
    X_next = next_month_df[features]

    y_pred_next = model.predict(X_next)[0]
    return next_month, y_pred_next

def predict_for_product(product_id):
    # Đọc dữ liệu đầu vào
    df = pd.read_csv("../data/data/processed_sales.csv")

    # Load cả 2 model
    model_rf_path = "../models/model_randomforest_sales.pkl"
    model_xgb_path = "../models/model_xgboost_sales.pkl"

    if not os.path.exists(model_rf_path) or not os.path.exists(model_xgb_path):
        print(f"Vui lòng train và lưu cả 2 model tại: {model_rf_path} và {model_xgb_path}")
        return

    model_rf = joblib.load(model_rf_path)
    model_xgb = joblib.load(model_xgb_path)

    # Kiểm tra dữ liệu cho product_id
    df_product = df[df['product_id'] == product_id]
    if df_product.empty:
        print(f"Không tìm thấy dữ liệu cho product_id={product_id}")
        return

    # Dự đoán theo tháng bằng 2 model
    monthly_features, preds_rf = predict_monthly_sales(model_rf, df, product_id)
    _, preds_xgb = predict_monthly_sales(model_xgb, df, product_id)

    # In kết quả dự đoán tháng gần nhất
    next_month, next_pred_rf = predict_next_month(model_rf, monthly_features)
    _, next_pred_xgb = predict_next_month(model_xgb, monthly_features)

    print(f"\nDự đoán số lượng bán cho product_id={product_id} tháng tiếp theo (tháng {next_month}):")
    print(f"- RandomForest: {next_pred_rf:.2f} sản phẩm")
    print(f"- XGBoost:      {next_pred_xgb:.2f} sản phẩm")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Vui lòng truyền product_id cần dự đoán.")
        sys.exit(1)

    try:
        pid = int(sys.argv[1])
    except ValueError:
        print("product_id phải là số nguyên.")
        sys.exit(1)

    predict_for_product(pid)
