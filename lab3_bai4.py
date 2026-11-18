# Bài 4: Quy trình ML hoàn chỉnh - Phần 3: Thử nghiệm mô hình Regression
# digits là bài toán phân loại nên bài này dùng tập california_housing (bài toán hồi quy):
# dự đoán giá nhà trung vị của mỗi khu vực ở California (đơn vị: 100.000 USD).
# Quy trình: Load -> Preprocess (Scaling) -> Train -> Evaluate (MSE) -> Visualize
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent


def load_data():
    housing = fetch_california_housing()  # lần đầu chạy sẽ tải dữ liệu từ Internet
    return housing.data, housing.target, housing.feature_names


def preprocess(X_train, X_test):
    # Scaler chỉ học mean/std từ tập train, sau đó áp dụng cho cả train và test
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def visualize(y_test, y_pred, n=300):
    # Vẽ một phần kết quả: n mẫu test đầu tiên, giá thật so với giá dự đoán
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test[:n], y_pred[:n], s=15, alpha=0.6)
    lim = [min(y_test[:n].min(), y_pred[:n].min()) - 0.2, max(y_test[:n].max(), y_pred[:n].max()) + 0.2]
    plt.plot(lim, lim, "r--", label="Dự đoán hoàn hảo (y = ŷ)")
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("Giá thật (100.000 USD)")
    plt.ylabel("Giá dự đoán (100.000 USD)")
    plt.title(f"California Housing - {n} mẫu test đầu tiên")
    plt.legend()
    plt.grid(alpha=0.3)
    path = OUTPUT_DIR / "california_thuc_te_vs_du_doan.png"
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


if __name__ == "__main__":
    # 1. Load
    X, y, feature_names = load_data()
    print("=== 1. Load dữ liệu ===")
    print("X:", X.shape, "| y:", y.shape)
    print("Đặc trưng:", feature_names)
    print(f"Giá nhà: từ {y.min():.2f} đến {y.max():.2f}, trung bình {y.mean():.2f} (100.000 USD)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Preprocess (Scaling)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    print("\n=== 2. Preprocess (StandardScaler) ===")
    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Std từng cột trước khi chuẩn hóa:", np.round(X_train.std(axis=0), 2).tolist())
    print("Std từng cột sau khi chuẩn hóa: ", np.round(X_train_scaled.std(axis=0), 2).tolist())

    # 3. Train
    model = LinearRegression().fit(X_train_scaled, y_train)
    print("\n=== 3. Train ===")
    print("Hệ số của từng đặc trưng (đã chuẩn hóa nên so sánh được độ ảnh hưởng):")
    for name, coef in sorted(zip(feature_names, model.coef_), key=lambda t: -abs(t[1])):
        print(f"  {name:<11} {coef:+.4f}")
    print(f"  Intercept   {model.intercept_:.4f}")

    # 4. Evaluate (MSE)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print("\n=== 4. Evaluate ===")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {np.sqrt(mse):.4f}  (~{np.sqrt(mse) * 100_000:,.0f} USD)")
    print(f"R²   = {r2_score(y_test, y_pred):.4f}")

    # 5. Visualize một phần kết quả
    print("\n=== 5. Visualize ===")
    print("Đã lưu biểu đồ:", visualize(y_test, y_pred).name)
    print("5 mẫu test đầu (giá thật -> dự đoán):")
    for yt, yp in zip(y_test[:5], y_pred[:5]):
        print(f"  {yt:.3f} -> {yp:.3f}")
