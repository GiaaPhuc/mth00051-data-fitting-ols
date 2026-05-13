# Data Fitting và Phương Pháp OLS

Đồ án 2 — Môn Toán Ứng Dụng và Thống Kê (MTH00051)  
Đại học Khoa học Tự nhiên — ĐHQG TP.HCM, HK2 2025–2026

---

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy notebook

```bash
jupyter notebook
```

Mở `part1/part1_notebook.ipynb` cho phần lý thuyết và `part2/part2_notebook.ipynb` cho phần ứng dụng thực tế.

---

## Cấu trúc thư mục

```
mth00051-data-fitting-ols/
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/                        # Lý thuyết và minh họa
│   ├── ols_implementation.py     # ols_fit, hat_matrix, model_metrics, coef_inference, vif
│   ├── ridge_lasso.py            # ridge_fit, lasso_fit
│   ├── residual_analysis.py      # residual_plots
│   ├── cross_validation.py       # kfold_cv
│   └── part1_notebook.ipynb
└── part2/                        # Ứng dụng dữ liệu thực
    ├── data/
    │   └── <ten_dataset>.csv
    ├── data_pipeline.py          # class DataPipeline
    ├── model_comparison.py       # compare_models
    ├── advanced_methods.py       # KernelRidgeRegression, BayesianLinearRegression
    └── part2_notebook.ipynb
```

---

## Dữ liệu

Đặt file CSV vào `part2/data/`. Nếu file quá lớn để commit, tải tại đây:

- **Nguồn:** _(điền link)_

---

## Thành viên nhóm

| Họ và tên | MSSV | Phân công |
| --------- | ---- | --------- |
|           |      |           |
|           |      |           |
