import pandas as pd
import os
import holidays

def preprocess_sales_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    input_file = os.path.join(base_dir, "../data/data/sales.csv")
    output_dir = os.path.join(base_dir, "../data/data")
    output_file = os.path.join(output_dir, "processed_sales.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # Trích xuất đặc trưng thời gian
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

    # Mã hóa category_name
    df['category_code'] = df['category_name'].astype('category').cat.codes

    # Thêm cột trend (theo thời gian tăng dần)
    df = df.sort_values('date')
    df['trend'] = range(1, len(df) + 1)

    # Ngày lễ ở Việt Nam
    vn_holidays = holidays.Vietnam(years=df['year'].unique())
    df['is_weekend'] = df['dayofweek'].isin([5, 6])
    df['is_official_holiday'] = df['date'].isin(vn_holidays)
    df['is_holiday'] = (df['is_weekend'] | df['is_official_holiday']).astype(int)

    # Xoá cột phụ
    df.drop(columns=['is_weekend', 'is_official_holiday'], inplace=True)

    # Ghi file
    df.to_csv(output_file, index=False)

    return f"✅ Đã xử lý và lưu dữ liệu tại {output_file}"

if __name__ == "__main__":
    result = preprocess_sales_data()
    print(result)