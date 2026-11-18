# Bài 3: Trực quan hóa kết quả
# Dùng lại mô hình của Bài 2 (diabetes, 1 đặc trưng BMI).
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lưu ảnh ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    diabetes = load_diabetes()
    X = diabetes.data[:, [diabetes.feature_names.index("bmi")]]
    y = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_train, y_train)

    # Đường hồi quy: dự đoán trên dải giá trị BMI từ nhỏ nhất đến lớn nhất
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, s=15, alpha=0.4, color="tab:blue", label="Dữ liệu train")
    plt.scatter(X_test, y_test, s=25, alpha=0.9, color="tab:orange", edgecolor="k", label="Dữ liệu test")
    plt.plot(x_line, y_line, color="tab:red", linewidth=2.5, label="Đường hồi quy (dự đoán)")
    plt.xlabel("BMI (đã chuẩn hóa)")
    plt.ylabel("Mức độ tiến triển bệnh")
    plt.title(f"Hồi quy tuyến tính: y = {model.coef_[0]:.1f} * BMI + {model.intercept_:.1f}")
    plt.legend()
    plt.grid(alpha=0.3)

    path = OUTPUT_DIR / "bieu_do_hoi_quy_diabetes.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print("Đã lưu biểu đồ:", path.name)
    print(
        "Nhận xét: đường hồi quy đi lên, BMI càng cao thì mức độ bệnh dự đoán càng cao."
        "\nCác điểm phân tán khá rộng quanh đường thẳng, cho thấy BMI chỉ giải thích được một phần của y."
    )
