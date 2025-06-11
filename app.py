from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys
import joblib
import pandas as pd

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data", "processed_sales.csv")
MODEL_RF_PATH = os.path.join(BASE_DIR, "models", "model_randomforest_monthly.pkl")
MODEL_XGB_PATH = os.path.join(BASE_DIR, "models", "model_xgboost_sales.pkl")

# Import các hàm train
sys.path.append(os.path.join(BASE_DIR, 'src'))
from train_random_forest import train_model as train_rf_model
from train_xgboost import train_xgboost_model

app = Flask(__name__)
CORS(app)

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không tìm thấy file tại: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def get_models(product_id):
    if not os.path.exists(MODEL_RF_PATH):
        train_rf_model(product_id)
    if not os.path.exists(MODEL_XGB_PATH):
        train_xgboost_model(product_id)
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
        # Load model và dữ liệu
        model_rf, model_xgb = get_models(product_id)
        df = load_data()
        df_product = df[df['product_id'] == product_id]

        if df_product.empty:
            return jsonify({"error": f"Không tìm thấy dữ liệu cho product_id={product_id}"}), 404

        # Gộp dữ liệu để huấn luyện và dự đoán (dựa theo tháng)
        monthly_features = df_product.groupby(['year', 'month']).agg({
            'day': 'mean',
            'dayofweek': 'mean',
            'price': 'mean',
            'discount_price': 'mean',
            'stock': 'mean',
            'sold': 'mean',
            'category_code': 'first',
            'trend': 'first',
            'is_holiday': 'max'
        }).reset_index()

        # Tính tổng quantity_sold theo từng tháng
        monthly_quantity = df_product.groupby(['year', 'month'])['quantity_sold'].sum().reset_index()
        monthly_quantity.rename(columns={'quantity_sold': 'total_quantity_sold'}, inplace=True)
        monthly_quantity['total_quantity_sold'] = monthly_quantity['total_quantity_sold'].round(2)

        # Các features đầu vào cho model
        features = [
            'year', 'month', 'day', 'dayofweek',
            'price', 'discount_price', 'stock', 'sold',
            'category_code', 'trend', 'is_holiday'
        ]

        # Dự đoán lại các tháng cũ (backtest)
        backtest_text_lines = []
        for i in range(len(monthly_features)):
            row = monthly_features.iloc[i]
            X_input = pd.DataFrame([row[features]])
            pred_rf = model_rf.predict(X_input)[0]
            pred_xgb = model_xgb.predict(X_input)[0]
            year, month = int(row['year']), int(row['month'])
            backtest_text_lines.append(
                f"  {year}-{month:02} | RF: {round(pred_rf, 2):<5} | XGB: {round(pred_xgb, 2):<5}"
            )

        # Dự đoán tháng tiếp theo
        last_row = monthly_features.iloc[-1]
        year, month = int(last_row['year']), int(last_row['month'])
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)

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
            'trend': last_row['trend'],
            'is_holiday': 0
        }

        next_month_df = pd.DataFrame([next_month_data])
        X_next = next_month_df[features]
        y_pred_rf = model_rf.predict(X_next)[0]
        y_pred_xgb = model_xgb.predict(X_next)[0]

        # Lịch sử sold thực tế
        history = monthly_features[['year', 'month', 'sold']].copy()
        history['sold'] = history['sold'].round(2)

        # Format log hiển thị console
        predictions_text = "\n".join(backtest_text_lines)
        next_text = (
            f"\n📈 Dự đoán tháng tiếp theo ({next_year}-{next_month:02}):\n"
            f" - Random Forest: {round(y_pred_rf, 2)} sản phẩm\n"
            f" - XGBoost:        {round(y_pred_xgb, 2)} sản phẩm"
        )
        print(predictions_text + next_text)

        # Trả kết quả JSON
        return jsonify({
            "product_id": product_id,
            "backtest_predictions": backtest_text_lines,
            "next_month_prediction": {
                "year": next_year,
                "month": next_month,
                "predicted_quantity_sold_random_forest": float(round(y_pred_rf, 2)),
                "predicted_quantity_sold_xgboost": float(round(y_pred_xgb, 2)),
            },
            "history": history.to_dict(orient="records"),
            "monthly_quantity_sold": [
                {
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "total_quantity_sold": float(row["total_quantity_sold"])
                }
                for _, row in monthly_quantity.iterrows()
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
