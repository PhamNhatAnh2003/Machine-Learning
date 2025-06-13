import requests
import pandas as pd
import os

def fetch_data_from_laravel(api_url="http://127.0.0.1:8000/api/sales-data", save_path=None):
    try:
        # Gọi API từ Laravel
        response = requests.get(api_url)
        response.raise_for_status()  # Báo lỗi nếu HTTP status không 200

        # Lấy dữ liệu JSON
        data = response.json()
        df = pd.DataFrame(data)

        # Xác định nơi lưu file
        if save_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, "data")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "sales.csv")

        # Ghi file
        df.to_csv(save_path, index=False)
        return f"✔ Dữ liệu đã được lưu vào {save_path}"

    except Exception as e:
        raise RuntimeError(f"Lỗi khi fetch dữ liệu từ Laravel: {str(e)}")
