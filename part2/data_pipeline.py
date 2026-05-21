import pandas as pd
import numpy as np


class DataPipeline:
    """
    Một pipeline tiền xử lý dữ liệu cho các mô hình Hồi quy Tuyến tính (OLS, Ridge, Lasso).
    Pipeline thực hiện các nhiệm vụ sau một cách tuần tự:
    1. Lọc và phân loại các biến liên tục (numerical) và biến phân loại (categorical).
    2. Xử lý dữ liệu khuyết thiếu (Missing Values Imputation).
    3. Mã hoá biến phân loại thành biến giả (One-Hot Encoding).
    4. Chuẩn hoá các biến liên tục (Z-score Standardization).

    Attributes:
        impute_num (str): Chiến lược điền khuyết cho biến số ('mean' hoặc 'median').
        impute_cat (str): Chiến lược điền khuyết cho biến phân loại ('mode').
        impute_values (dict): Từ điển lưu trữ các giá trị điền khuyết đã học.
        num_features (list): Danh sách tên các cột chứa dữ liệu số.
        cat_features (list): Danh sách tên các cột chứa dữ liệu phân loại.
        scaling_params (dict): Từ điển lưu trữ giá trị Mean và Std của từng cột số.
        dummy_columns_ (list): Danh sách tên tất cả các cột sau khi One-Hot Encoding trên
            tập train. Dùng để căn chỉnh (align) tập test về cùng cấu trúc cột.
    """

    def __init__(self, impute_num='mean', impute_cat='mode'):
        self.impute_num = impute_num
        self.impute_cat = impute_cat
        self.impute_values = {}
        self.num_features = []
        self.cat_features = []
        self.scaling_params = {}
        self.dummy_columns_ = []  # Được gán sau khi fit()

    def fit(self, X: pd.DataFrame, y=None):
        """
        Khảo sát tập huấn luyện (Train set) để tính toán và lưu trữ các tham số 
        cần thiết (giá trị thay thế, trung bình, độ lệch chuẩn).

        Args:
            X (pd.DataFrame): Tập dữ liệu huấn luyện.
            y (ignored): Bỏ qua (chỉ giữ lại để đồng nhất API theo chuẩn scikit-learn).

        Returns:
            self: Trả về chính đối tượng DataPipeline sau khi đã lưu thông số.
        """
        # Phân loại biến liên tục và biến phân loại
        self.num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

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
                'std': X[col].std()
            }

        # Ghi nhớ cấu trúc cột sau One-Hot Encoding trên tập train.
        # Mục đích: đảm bảo tập test luôn có cùng tập cột với tập train
        # (xử lý trường hợp test set có category mới hoặc thiếu category).
        X_dummy = pd.get_dummies(X, columns=self.cat_features, drop_first=True)
        self.dummy_columns_ = X_dummy.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng các tham số đã học từ hàm `fit` để biến đổi một tập dữ liệu bất kỳ.

        Args:
            X (pd.DataFrame): Tập dữ liệu cần được biến đổi (Train hoặc Test set).

        Returns:
            pd.DataFrame: Tập dữ liệu đã được làm sạch, mã hoá và chuẩn hoá hoàn chỉnh.
        """
        # Bước 1: Tạo bản sao để bảo toàn dữ liệu gốc
        X_transformed = X.copy()

        # Bước 2: Điền khuyết dữ liệu
        X_transformed = X_transformed.fillna(self.impute_values)

        # Bước 3: Categorical Encoding (One-Hot Encoding)
        # Tham số drop_first=True được sử dụng để loại bỏ bớt một biến giả,
        # qua đó ngăn chặn hiện tượng đa cộng tuyến hoàn hảo (Perfect Multicollinearity).
        #
        # ⚠️  Hạn chế đã biết của pd.get_dummies():
        #   pd.get_dummies() tạo cột dựa trên các category *thực sự xuất hiện* trong
        #   tập dữ liệu được truyền vào. Nếu tập test chứa category chưa từng thấy
        #   trong tập train, get_dummies sẽ tạo ra các cột dư thừa → số cột lệch nhau
        #   → mô hình báo lỗi khi predict. Ngược lại, nếu tập test thiếu một category,
        #   cột tương ứng sẽ không được tạo ra → cũng gây lệch cột.
        #
        #   Giải pháp triệt để hơn là dùng sklearn.preprocessing.OneHotEncoder
        #   (lưu mapping từ fit, áp dụng nhất quán khi transform). Tuy nhiên, với
        #   bộ dữ liệu này (train/test cùng phân phối, split ngẫu nhiên), rủi ro
        #   xuất hiện category mới là rất thấp. Ta xử lý bằng cách căn chỉnh cột
        #   theo danh sách đã lưu từ fit(): cột thừa bị loại, cột thiếu được điền 0.
        X_transformed = pd.get_dummies(
            X_transformed, columns=self.cat_features, drop_first=True
        )

        # Căn chỉnh cột về đúng cấu trúc của tập train:
        #   - fill_value=0: cột thiếu (category không xuất hiện trong test) → điền 0
        #   - Cột thừa (category mới trong test)          → tự động bị loại bỏ
        X_transformed = X_transformed.reindex(
            columns=self.dummy_columns_, fill_value=0
        )

        # Bước 4: Chuẩn hoá Z-score cho biến liên tục
        for col in self.num_features:
            mean = self.scaling_params[col]['mean']
            std = self.scaling_params[col]['std']
            
            # Ngăn chặn lỗi chia cho 0 trong trường hợp cột có phương sai bằng 0
            X_transformed[col] = (X_transformed[col] - mean) / std if std != 0 else 0.0

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Tiện ích gộp chung hai bước fit và transform.
        Thường được gọi duy nhất 1 lần trên tập Train.

        Args:
            X (pd.DataFrame): Tập dữ liệu huấn luyện.
            y (ignored): Bỏ qua.

        Returns:
            pd.DataFrame: Tập dữ liệu huấn luyện đã được biến đổi.
        """
        self.fit(X, y)
        return self.transform(X)
