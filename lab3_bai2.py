# Bài 2: Đánh giá mô hình
#
# Ý nghĩa các chỉ số đánh giá (y là giá trị thật, ŷ là giá trị dự đoán, n là số mẫu):
#
# - MAE (Mean Absolute Error) = (1/n) * Σ|y - ŷ|
#   Sai số tuyệt đối trung bình: trung bình mỗi dự đoán lệch bao nhiêu so với thực tế,
#   cùng đơn vị với y nên dễ hiểu. Càng nhỏ càng tốt.
#
# - MSE (Mean Squared Error) = (1/n) * Σ(y - ŷ)²
#   Sai số bình phương trung bình: bình phương nên phạt nặng các dự đoán lệch nhiều,
#   nhạy với ngoại lệ. Đơn vị là bình phương đơn vị của y nên khó diễn giải trực tiếp;
#   thường lấy căn (RMSE) để đưa về cùng đơn vị. Càng nhỏ càng tốt.
#
# - R-squared (R², hệ số xác định) = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
#   Tỉ lệ biến thiên của y mà mô hình giải thích được.
#   R² = 1: dự đoán hoàn hảo; R² = 0: không tốt hơn việc luôn đoán bằng giá trị trung bình;
#   R² < 0: còn tệ hơn đoán trung bình.
import sys

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

FEATURE = "bmi"  # chỉ số khối cơ thể - đặc trưng tương quan mạnh nhất với y


def load_data():
    diabetes = load_diabetes()
    idx = diabetes.feature_names.index(FEATURE)
    X = diabetes.data[:, [idx]]  # giữ dạng 2 chiều (n, 1)
    y = diabetes.target  # mức độ tiến triển bệnh sau 1 năm
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"=== Tập diabetes - chỉ dùng đặc trưng '{FEATURE}' ===")
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nPhương trình: y = {model.coef_[0]:.2f} * {FEATURE} + {model.intercept_:.2f}")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Chỉ số đánh giá trên tập Test ===")
    print(f"MAE  = {mae:.2f}")
    print(f"MSE  = {mse:.2f}  (RMSE = {np.sqrt(mse):.2f})")
    print(f"R²   = {r2:.4f}")
    print(f"(Tham khảo: y trong tập test có trung bình {y_test.mean():.2f}, độ lệch chuẩn {y_test.std():.2f})")

    print(
        f"\nNhận xét: trung bình mỗi dự đoán lệch khoảng {mae:.0f} đơn vị so với thực tế."
        f"\nR² = {r2:.2f} nghĩa là chỉ riêng BMI giải thích được khoảng {r2:.0%} sự biến thiên của"
        "\nmức độ bệnh; phần còn lại phụ thuộc vào các yếu tố khác mà mô hình 1 đặc trưng không thấy."
    )
