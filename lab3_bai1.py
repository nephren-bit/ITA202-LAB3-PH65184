# Bài 1: Xây dựng mô hình Hồi quy tuyến tính
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")


def tao_du_lieu(n=100, seed=42):
    # Dữ liệu giả lập: y = 2X + 1 + nhiễu (nhiễu phân phối chuẩn, độ lệch chuẩn 1)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 1))  # sklearn yêu cầu X dạng 2 chiều (n mẫu, 1 đặc trưng)
    y = 2 * X[:, 0] + 1 + rng.normal(0, 1, size=n)
    return X, y


if __name__ == "__main__":
    X, y = tao_du_lieu()
    print("=== Dữ liệu giả lập: y = 2X + 1 + nhiễu ===")
    print("Shape X:", X.shape, "| Shape y:", y.shape)
    print("5 mẫu đầu (X, y):")
    for xi, yi in zip(X[:5, 0], y[:5]):
        print(f"  X = {xi:.3f}  ->  y = {yi:.3f}")

    model = LinearRegression()
    model.fit(X, y)

    print("\n=== Hệ số của mô hình ===")
    print(f"Coefficient (hệ số góc): {model.coef_[0]:.4f}   (giá trị thật: 2)")
    print(f"Intercept (hệ số chặn):  {model.intercept_:.4f}   (giá trị thật: 1)")
    print(f"Phương trình học được: y = {model.coef_[0]:.4f}X + {model.intercept_:.4f}")

    X_moi = np.array([[0], [5], [10]])
    print("\nDự đoán với X mới:")
    for xi, yi in zip(X_moi[:, 0], model.predict(X_moi)):
        print(f"  X = {xi:>2}  ->  y dự đoán = {yi:.3f}  (theo công thức thật: {2 * xi + 1})")

    print("\nNhận xét: hệ số học được gần với 2 và 1; sai lệch nhỏ là do nhiễu ngẫu nhiên trong dữ liệu.")
