<div align="center">

# Data Fitting & Phương Pháp OLS

**Đồ án 2 — MTH00051: Toán Ứng Dụng và Thống Kê**  
Khoa Công nghệ Thông tin — Trường ĐH Khoa học Tự nhiên — ĐHQG TP.HCM · HK2 2025–2026

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.12+-8CAAE6?logo=scipy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## Giới thiệu

Đồ án tập trung vào hai nhóm nhiệm vụ:

1. **Lý thuyết và minh họa** — Nắm vững nền tảng toán học của data fitting và phương pháp Ordinary Least Squares (OLS), minh họa bằng code Python tự cài đặt từ đầu.
2. **Ứng dụng thực tế** — Vận dụng data fitting để phân tích bộ dữ liệu thực, bao gồm tiền xử lý, xây dựng mô hình hồi quy và đánh giá kết quả có hệ thống.

> **Lưu ý:** Các hàm như `sklearn.linear_model.LinearRegression`, `numpy.linalg.lstsq` chỉ được dùng để **kiểm chứng**. Toàn bộ thuật toán chính phải được cài đặt từ đầu dựa trên công thức toán học.

---

## Thông tin môn học

| Thông tin    | Chi tiết                                 |
| ------------ | ---------------------------------------- |
| Môn học      | Toán Ứng Dụng và Thống Kê — MTH00051     |
| GV Thực hành | ThS. Võ Nam Thục Đoan · ThS. Lê Nhựt Nam |
| Email GV     | {vntdoan, lnnam}@fit.hcmus.edu.vn        |
| Hạn nộp bài  | **30/05/2026, trước 23:59**              |
| Nộp qua      | Moodle của Khoa                          |

---

## Tổng quan hai phần

| Phần       | Nội dung                                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| **Part 1** | Xây dựng thư viện OLS từ đầu: ước lượng hệ số, hat matrix, kiểm định thống kê, VIF, kèm bộ unit test đầy đủ |
| **Part 2** | Áp dụng OLS và các phương pháp mở rộng (Ridge, Lasso, Kernel Ridge, Bayesian) trên bộ dữ liệu thực tế       |

---

## Cấu trúc thư mục

```
mth00051-data-fitting-ols/
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── automobile/                       # Bộ dữ liệu UCI Automobile
│   ├── imports-85.data
│   └── imports-85.names
├── part1/                            # Lý thuyết & minh họa
│   ├── ols_implementation.py         # ols_fit, hat_matrix, model_metrics,
│   │                                 #   coef_inference, vif
│   ├── ridge_lasso.py                # ridge_fit, lasso_fit
│   ├── residual_analysis.py          # residual_plots
│   ├── cross_validation.py           # kfold_cv
│   ├── test_ols_implementation.py    # Unit tests (≥ 2 tests / hàm)
│   └── part1_notebook.ipynb
└── part2/                            # Ứng dụng dữ liệu thực
    ├── data/
    │   └── <ten_dataset>.csv
    ├── data_pipeline.py              # class DataPipeline
    ├── model_comparison.py           # compare_models
    ├── advanced_methods.py           # KernelRidgeRegression,
    │                                 # BayesianLinearRegression
    └── part2_notebook.ipynb
```

---

## Cài đặt

**Yêu cầu:** Python ≥ 3.10

```bash
# Clone repository
git clone <repo-url>
cd mth00051-data-fitting-ols

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## Sử dụng

### Chạy Jupyter Notebook

```bash
jupyter notebook
```

Mở `part1/part1_notebook.ipynb` (lý thuyết & minh họa) hoặc `part2/part2_notebook.ipynb` (ứng dụng thực tế).

### Chạy Unit Tests

```bash
python part1/test_ols_implementation.py
```

---

## Mô tả các module

### Part 1

| File                         | Nội dung chính                                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ols_implementation.py`      | `ols_fit` — Normal Equations; `hat_matrix` — H = QQᵀ; `model_metrics` — R², F-test; `coef_inference` — t-test, CI; `vif` — phát hiện đa cộng tuyến |
| `ridge_lasso.py`             | `ridge_fit` — Ridge Regression (L2), vẽ ridge trace; `lasso_fit` — Lasso Regression (L1) qua coordinate descent                                    |
| `residual_analysis.py`       | `residual_plots` — 4 biểu đồ chẩn đoán phần dư (Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance)                                         |
| `cross_validation.py`        | `kfold_cv` — k-fold cross-validation, tính CV score, so sánh mô hình theo AIC/BIC                                                                  |
| `test_ols_implementation.py` | Unit tests (≥ 2 test / hàm) kiểm chứng kết quả trên dữ liệu đã biết                                                                                |

### Part 2

| File                  | Nội dung chính                                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `data_pipeline.py`    | `class DataPipeline` — EDA, xử lý missing values, encoding, chuẩn hóa; `fit` trên train, `transform` trên test       |
| `model_comparison.py` | So sánh ≥ 3 mô hình (OLS, Ridge/Lasso, mô hình khác) — bảng MAE, RMSE, R² trên test set                              |
| `advanced_methods.py` | `KernelRidgeRegression` — kernel trick (RBF, polynomial); `BayesianLinearRegression` — posterior, credible intervals |

---

## Dữ liệu

Đồ án sử dụng bộ dữ liệu **[UCI Automobile](https://archive.ics.uci.edu/dataset/10/automobile)** (`automobile/imports-85.data`), gồm 205 quan sát và 26 thuộc tính mô tả đặc điểm kỹ thuật, bảo hiểm và giá xe hơi năm 1985.

Tiêu chí bộ dữ liệu Part 2: thực tế (real-world), có missing values (≥ 5%), biến mục tiêu liên tục, n ≥ 200 quan trắc, p ≥ 3 biến đặc trưng, từ nguồn đáng tin cậy (Kaggle, UCI, data.gov, v.v.).

---

## Thành viên nhóm

| Họ và tên               | MSSV     | Phân công |
| ----------------------- | -------- | --------- |
| Trương Bảo Nguyên       | 24120397 |           |
| Đặng Quang Huy          | 24120322 |           |
| Trần Nguyễn Lâm Gia Thụ | 24120458 |           |
| Nguyễn Khánh Đăng       | 24120171 |           |
| Võ Gia Phúc             | 24120413 |           |

---
