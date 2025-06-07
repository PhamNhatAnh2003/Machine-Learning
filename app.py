from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys
import joblib
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_random_forest import train_model as train_rf_model  # Hàm train model RandomForest
from train_random_forest import train_model as train_rf_model
from train_xgboost import train_xgboost_model
    # Hàm train model XGBoost

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data", "processed_sales.csv")
MODEL_RF_PATH = os.path.join(BASE_DIR, "models", "model_randomforest_sales.pkl")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "models", "model_xgboost_sales.pkl")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file tại: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def get_models(product_id):
    # Nếu chưa có model, train lại
    if not os.path.exists(MODEL_RF_PATH):
        train_rf_model(product_id)
    if not os.path.exists(MODEL_XGB_PATH):
        train_xgboost_model(product_id)
    # Luôn tải model từ file
    model_rf = joblib.load(MODEL_RF_PATH)
    model_xgb = joblib.load(MODEL_XGB_PATH)
    return model_rf, model_xgb

@app.route('/train/<int:product_id>')
def train(product_id):
    try:
        train_rf_model(product_id)
        train_xgboost_model(product_id)
        return jsonify({"message": f"✔ Đã train model RandomForest và XGBoost cho product_id={product_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<int:product_id>')
def predict(product_id):
    try:
        model_rf, model_xgb = get_models(product_id)
        df = load_data()
        df_product = df[df['product_id'] == product_id]
        if df_product.empty:
            return jsonify({"error": f"Không tìm thấy dữ liệu cho product_id={product_id}"}), 404

        # Chuẩn bị dữ liệu nhóm theo tháng
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

        # Tạo dữ liệu tháng tiếp theo
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
        X_next = next_month_df[features]

        y_pred_rf = model_rf.predict(X_next)[0]
        y_pred_xgb = model_xgb.predict(X_next)[0]
 # Lịch sử bán hàng theo tháng
        history = df_product.groupby(['year', 'month']).agg({
            'sold': 'sum'
        }).reset_index().sort_values(['year', 'month'])

        return jsonify({
            "product_id": product_id,
            "next_month_prediction": {
                "year": next_year,
                "month": next_month,
                "predicted_quantity_sold_random_forest": float(round(y_pred_rf, 2)),
                "predicted_quantity_sold_xgboost": float(round(y_pred_xgb, 2)),
            },
            "history": history.to_dict(orient="records")  # 👈 Thêm dòng này
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
