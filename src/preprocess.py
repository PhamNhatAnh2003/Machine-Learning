import pandas as pd
import os

# Đường dẫn file
input_file = "../data/data/sales.csv"  # Giả sử bạn chạy từ thư mục src
output_dir = "../data/data"
output_file = os.path.join(output_dir, "processed_sales.csv")

# Tạo thư mục nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(input_file)

# Chuyển cột date sang kiểu datetime
df['date'] = pd.to_datetime(df['date'])

# Trích xuất đặc trưng thời gian
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Mã hóa category_name
df['category_code'] = df['category_name'].astype('category').cat.codes

# Hiển thị dữ liệu
print("Dữ liệu sau tiền xử lý (5 dòng đầu):")
print(df.head())

# Ghi file kết quả
df.to_csv(output_file, index=False)
print(f"\n Đã lưu file vào {output_file}")
