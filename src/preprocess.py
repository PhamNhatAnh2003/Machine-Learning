import pandas as pd
import os
import holidays

# Đường dẫn file
input_file = "../data/data/sales.csv"
output_dir = "../data/data"
output_file = os.path.join(output_dir, "processed_sales.csv")

# Tạo thư mục nếu chưa có
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

# Thêm cột trend (tăng dần theo thời gian)
df = df.sort_values('date')
df['trend'] = range(1, len(df) + 1)

# Tạo danh sách ngày lễ chính thức của Việt Nam
vn_holidays = holidays.Vietnam(years=df['year'].unique())

# Tính toán is_holiday = 1 nếu là cuối tuần hoặc ngày lễ
df['is_weekend'] = df['dayofweek'].isin([5, 6])
df['is_official_holiday'] = df['date'].isin(vn_holidays)
df['is_holiday'] = (df['is_weekend'] | df['is_official_holiday']).astype(int)

# (Tuỳ chọn) Xoá cột phụ nếu không cần
df.drop(columns=['is_weekend', 'is_official_holiday'], inplace=True)

# Hiển thị dữ liệu mẫu
print("Dữ liệu sau tiền xử lý (5 dòng đầu):")
print(df.head())

# Ghi file kết quả
df.to_csv(output_file, index=False)
print(f"\n✅ Đã lưu file vào {output_file}")
