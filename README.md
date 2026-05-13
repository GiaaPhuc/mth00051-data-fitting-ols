<div align="center">

# 📐 Data Fitting & Phương Pháp OLS

**Đồ án 2 — MTH00051: Toán Ứng Dụng và Thống Kê**  
Trường Đại học Khoa học Tự nhiên — ĐHQG TP.HCM · HK2 2025–2026

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.12+-8CAAE6?logo=scipy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## Tổng quan

Đồ án triển khai **Ordinary Least Squares (OLS)** từ nền tảng toán học, không dựa vào các hàm hồi quy có sẵn của `scikit-learn`. Dự án được chia làm hai phần:

| Phần       | Nội dung                                                                                                    |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| **Part 1** | Xây dựng thư viện OLS từ đầu: ước lượng hệ số, hat matrix, kiểm định thống kê, VIF, kèm bộ unit test đầy đủ |
| **Part 2** | Áp dụng OLS và các phương pháp mở rộng (Ridge, Lasso, Kernel Ridge, Bayesian) trên bộ dữ liệu thực tế       |

---

## Cấu trúc thư mục

```
mth00051-data-fitting-ols/
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
    ├── data_pipeline.py              # class DataPipeline
    ├── model_comparison.py           # compare_models
    ├── advanced_methods.py           # KernelRidgeRegression,
    │                                 # BayesianLinearRegression
    └── part2_notebook.ipynb
```

---

## Cài đặt

**Yêu cầu:** Python ≥ 3.11

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

## API — `part1/ols_implementation.py`

Tất cả hàm nhận **ma trận design đã có cột intercept** `X` shape `(n, p+1)`.

| Hàm                                      | Đầu vào       | Đầu ra                                      | Mô tả                                                       |
| ---------------------------------------- | ------------- | ------------------------------------------- | ----------------------------------------------------------- |
| `ols_fit(X, y)`                          | `X`, `y`      | `beta_hat, sigma2, y_hat, residuals, rss`   | Ước lượng OLS qua Normal Equations: β̂ = (XᵀX)⁻¹Xᵀy          |
| `hat_matrix(X)`                          | `X`           | `H` (n×n)                                   | Hat matrix H = QQᵀ (QR decomposition); kiểm tra 4 tính chất |
| `model_metrics(y, y_hat, p)`             | `y`, `ŷ`, `p` | `r2, r2_adj, f_stat, f_pvalue, ...`         | R², R² hiệu chỉnh, F-test tổng thể                          |
| `coef_inference(X, y, beta_hat, sigma2)` | —             | `se, t_stats, p_values, ci_lower, ci_upper` | Kiểm định t và khoảng tin cậy cho từng hệ số                |
| `vif(X)`                                 | `X`           | `vif_values` (p,)                           | VIF phát hiện đa cộng tuyến; VIF > 10 → đáng lo ngại        |

---

## Dữ liệu

Đồ án sử dụng bộ dữ liệu **[UCI Automobile](https://archive.ics.uci.edu/dataset/10/automobile)** (`automobile/imports-85.data`), gồm 205 quan sát và 26 thuộc tính mô tả đặc điểm kỹ thuật, bảo hiểm và giá xe hơi năm 1985.

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
