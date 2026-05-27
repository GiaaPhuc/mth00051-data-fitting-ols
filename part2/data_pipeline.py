import pandas as pd
import numpy as np


class DataPipeline:
    """.
    Pipeline thực hiện các nhiệm vụ sau một cách tuần tự:
    1. Lọc và phân loại các biến liên tục (numerical) và biến phân loại (categorical).
    2. Xử lý dữ liệu khuyết thiếu (Missing Values Imputation).
    3. Mã hoá biến phân loại thành biến giả (One-Hot Encoding).
    4. Chuẩn hoá các biến liên tục (Z-score Standardization).
    5. Loại bỏ cột có phương sai quá thấp (Variance Threshold).

    Attributes:
        impute_num (str): điền khuyết cho biến số ('mean' hoặc 'median').
        impute_cat (str): điền khuyết cho biến phân loại ('mode').
        var_threshold (float): Ngưỡng phương sai tối thiểu.
        impute_values (dict): Lưu trữ các giá trị điền khuyết đã học.
        num_features (list): Danh sách tên các cột chứa dữ liệu số.
        cat_features (list): Danh sách tên các cột chứa dữ liệu phân loại.
        scaling_params (dict): Lưu trữ giá trị Mean và Std của từng cột số.
        dummy_columns_ (list): Danh sách tên tất cả các cột sau khi One-Hot Encoding trên
            tập train. Dùng để căn chỉnh (align) tập test về cùng cấu trúc cột.
        kept_columns_ (list): Danh sách cột còn lại sau khi lọc phương sai thấp.
    """

    def __init__(self, impute_num='mean', impute_cat='mode', var_threshold=1e-4):
        self.impute_num = impute_num
        self.impute_cat = impute_cat
        self.var_threshold = var_threshold   # Loại cột có phương sai < ngưỡng này
        self.impute_values = {}
        self.num_features = []
        self.cat_features = []
        self.scaling_params = {}
        self.dummy_columns_ = []  # Được gán sau khi fit()
        self.kept_columns_ = []  # Cột còn lại sau khi lọc phương sai

    def fit(self, X: pd.DataFrame, y=None):
        """
        Khảo sát tập huấn luyện để tính toán và lưu trữ các tham số
        cần thiết.

        Args:
            X (pd.DataFrame): Tập dữ liệu huấn luyện.
            y (ignored): Bỏ qua (chỉ giữ lại để đồng nhất API theo chuẩn scikit-learn).

        Returns:
            self: Trả về chính đối tượng DataPipeline sau khi đã lưu thông số.
        """
        # Phân loại biến liên tục và biến phân loại
        self.num_features = X.select_dtypes(
            include=[np.number]).columns.tolist()
        self.cat_features = X.select_dtypes(
            exclude=[np.number]).columns.tolist()

        # Tính toán giá trị điền khuyết cho biến liên tục
        for col in self.num_features:
            if self.impute_num == 'mean':
                self.impute_values[col] = X[col].mean()
            elif self.impute_num == 'median':
                self.impute_values[col] = X[col].median()

        # Tính toán giá trị điền khuyết cho biến phân loại
        for col in self.cat_features:
            if self.impute_cat == 'mode':
                self.impute_values[col] = X[col].mode()[0]

        # Tính toán các tham số chuẩn hoá (Mean, Std)
        for col in self.num_features:
            self.scaling_params[col] = {
                'mean': X[col].mean(),
                'std':  X[col].std()
            }

        X_filled = X.fillna(self.impute_values)

        # Ghi nhớ cấu trúc cột sau One-Hot Encoding trên tập train.
        # Đảm bảo tập test luôn có cùng tập cột với tập train
        # (xử lý trường hợp test set có category mới hoặc thiếu category).
        X_dummy = pd.get_dummies(
            X_filled, columns=self.cat_features, drop_first=True)
        self.dummy_columns_ = X_dummy.columns.tolist()

        # Lọc cột có phương sai quá thấp (gần hằng số) để tránh ma trận X^TX suy biến.
        # Áp dụng cho toàn bộ cột sau OHE (cả số lẫn biến giả).
        col_vars = X_dummy.var()
        self.kept_columns_ = col_vars[col_vars >=
                                      self.var_threshold].index.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng các tham số đã học từ hàm `fit` để biến đổi tập dữ liệu.

        Args:
            X (pd.DataFrame): Tập dữ liệu cần được biến đổi.

        Returns:
            pd.DataFrame: Tập dữ liệu đã được làm sạch, mã hoá và chuẩn hoá hoàn chỉnh.
        """
        # Tạo bản sao để bảo toàn dữ liệu gốc
        X_transformed = X.copy()

        # Điền khuyết dữ liệu
        X_transformed = X_transformed.fillna(self.impute_values)

        # Categorical Encoding (One-Hot Encoding)
        # Tham số drop_first=True được sử dụng để loại bỏ bớt một biến giả,
        # qua đó ngăn chặn hiện tượng đa cộng tuyến hoàn hảo (Perfect Multicollinearity).
        X_transformed = pd.get_dummies(
            X_transformed, columns=self.cat_features, drop_first=True
        )

        # Căn chỉnh cột về đúng cấu trúc của tập train:
        #   - fill_value=0: cột thiếu (category không xuất hiện trong test) → điền 0
        #   - Cột thừa (category mới trong test) → tự động bị loại bỏ
        X_transformed = X_transformed.reindex(
            columns=self.dummy_columns_, fill_value=0
        )

        # Chuẩn hoá Z-score cho biến liên tục
        # Chuẩn hóa TRƯỚC khi lọc kept_columns_ ---
        # Chỉ chuẩn hóa những cột số còn tồn tại trong X_transformed.
        for col in self.num_features:
            if col not in X_transformed.columns:
                continue
            mean = self.scaling_params[col]['mean']
            std = self.scaling_params[col]['std']

            X_transformed[col] = np.where(
                std != 0,
                (X_transformed[col] - mean) / std,
                0.0
            )

        # Loại bỏ các cột có phương sai quá thấp (đã xác định trong fit).
        if self.kept_columns_:
            X_transformed = X_transformed[self.kept_columns_]

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Gộp chung hai bước fit và transform.

        Args:
            X (pd.DataFrame): Tập dữ liệu huấn luyện.
            y (ignored): Bỏ qua.

        Returns:
            pd.DataFrame: Tập dữ liệu huấn luyện đã được biến đổi.
        """
        self.fit(X, y)
        return self.transform(X)
