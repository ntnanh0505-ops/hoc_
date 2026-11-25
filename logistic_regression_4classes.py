"""
========================================================================================
ĐỀ TÀI: PHÂN LOẠI 4 GIỐNG LÚA BẰNG THUẬT TOÁN HỒI QUY LOGISTIC (LOGISTIC REGRESSION)
========================================================================================
- Mục 2.5.1: Xây dựng mô hình Hồi quy Logistic bằng NumPy (tự cài đặt Softmax Regression từ đầu)
- Mục 2.5.2: Xây dựng mô hình Hồi quy Logistic bằng Scikit-learn
- Mục 2.5.3: Đánh giá & So sánh chi tiết hiệu năng giữa 2 phương pháp
- Mục 2.5.4: Trực quan hóa đường cong hàm mất mát (Loss Curve) và Ma trận nhầm lẫn (Confusion Matrix)

Dữ liệu: Rice_4Classes_Scaled_Splitted.xlsx (4 giống lúa: Cammeo, Gonen, Jasmine, Osmancik)
========================================================================================
"""

import sys
import os
import subprocess
import time

# Đảm bảo xuất chuẩn ký tự UTF-8 trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ======================================================================================
# 1. KIỂM TRA VÀ CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT
# ======================================================================================
def check_and_install_dependencies():
    """Tự động kiểm tra và cài đặt các thư viện cần thiết nếu chưa có."""
    required_packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn"
    }
    
    print("-" * 80)
    print("[BƯỚC 1] KIỂM TRA VÀ CÀI ĐẶT MÔI TRƯỜNG THƯ VIỆN")
    print("-" * 80)
    
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
            print(f" [OK] Thư viện '{pip_name}' đã được cài đặt sẵn.")
        except ImportError:
            print(f" [..] Đang cài đặt thư viện thiếu: '{pip_name}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f" [OK] Cài đặt thành công '{pip_name}'.")

check_and_install_dependencies()

# Import các thư viện sau khi đã kiểm tra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Thiết lập hiển thị tiếng Việt và đồ thị đẹp
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False


# ======================================================================================
# 2. ĐỌC DỮ LIỆU ĐẦU VÀO VÀ TIỀN XỬ LÝ (LABEL ENCODING)
# ======================================================================================
def load_and_preprocess_data():
    """
    Đọc dữ liệu từ file Excel Rice_4Classes_Scaled_Splitted.xlsx.
    Kiểm tra cấu trúc các sheet (Train, Val, Test) và mã hóa nhãn 4 giống lúa.
    """
    print("\n" + "-" * 80)
    print("[BƯỚC 2] ĐỌC DỮ LIỆU VÀ TIỀN XỬ LÝ NHÃN (LABEL ENCODING)")
    print("-" * 80)
    
    possible_paths = [
        r"C:\Users\LENOVO\Documents\Nam_cuoi\hoc_may\archive (1)\dataset\Rice_4Classes_Scaled_Splitted.xlsx",
        os.path.join(os.path.dirname(__file__), "archive (1)", "dataset", "Rice_4Classes_Scaled_Splitted.xlsx"),
        os.path.join(os.path.dirname(__file__), "Rice_4Classes_Scaled_Splitted.xlsx"),
        "Rice_4Classes_Scaled_Splitted.xlsx"
    ]
    
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_path = os.path.abspath(p)
            break
            
    if dataset_path is None:
        raise FileNotFoundError("Không tìm thấy file Rice_4Classes_Scaled_Splitted.xlsx trong các đường dẫn quy định!")
        
    print(f" Đang đọc dữ liệu từ: {dataset_path}")
    excel_file = pd.ExcelFile(dataset_path)
    sheet_names = excel_file.sheet_names
    print(f" Danh sách các sheet trong file Excel: {sheet_names}")
    
    # Xác định sheet Train, Test, Validation
    train_sheet = next((s for s in sheet_names if 'train' in s.lower()), None)
    val_sheet = next((s for s in sheet_names if 'val' in s.lower()), None)
    test_sheet = next((s for s in sheet_names if 'test' in s.lower()), None)
    
    if train_sheet and test_sheet:
        print(f"  -> Phát hiện cấu trúc chia sẵn:")
        print(f"     * Sheet Train : '{train_sheet}'")
        if val_sheet:
            print(f"     * Sheet Val   : '{val_sheet}'")
        print(f"     * Sheet Test  : '{test_sheet}'")
        
        train_df = excel_file.parse(train_sheet)
        test_df = excel_file.parse(test_sheet)
        val_df = excel_file.parse(val_sheet) if val_sheet else None
    else:
        print("  -> File không có sheet Train/Test riêng biệt, tiến hành phân chia Stratified 80/20...")
        df = excel_file.parse(sheet_names[0])
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
        val_df = None
        
    # Xác định cột nhãn và cột đặc trưng
    label_col = 'Class' if 'Class' in train_df.columns else train_df.columns[-1]
    feature_cols = [c for c in train_df.columns if c != label_col]
    
    print(f"\n Danh sách 7 đặc trưng hình học (Features):")
    for idx, feat in enumerate(feature_cols, 1):
        print(f"   {idx}. {feat}")
    print(f" Cột mục tiêu (Target/Label): '{label_col}'")
    
    # Mã hóa nhãn (Label Encoding)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[label_col])
    y_test = label_encoder.transform(test_df[label_col])
    
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    
    X_val, y_val = None, None
    if val_df is not None:
        y_val = label_encoder.transform(val_df[label_col])
        X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    
    class_names = list(label_encoder.classes_)
    print(f"\n Bảng mã hóa nhãn 4 giống lúa (Label Encoding):")
    for idx, name in enumerate(class_names):
        count_train = int(np.sum(y_train == idx))
        count_test = int(np.sum(y_test == idx))
        print(f"   - Lớp {idx}: '{name:<10}' | Train: {count_train:>5} mẫu | Test: {count_test:>5} mẫu")
        
    print(f"\n Kích thước dữ liệu:")
    print(f"   - Tập Huấn luyện (Train): X = {X_train.shape}, y = {y_train.shape}")
    if X_val is not None:
        print(f"   - Tập Xác thực   (Val)  : X = {X_val.shape}, y = {y_val.shape}")
    print(f"   - Tập Kiểm thử   (Test) : X = {X_test.shape}, y = {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, feature_cols


# ======================================================================================
# 3. MỤC 2.5.1: XÂY DỰNG MÔ HÌNH HỒI QUY LOGISTIC BẰNG NUMPY (SOFTMAX REGRESSION)
# ======================================================================================
class SoftmaxRegressionNumPy:
    """
    Mô hình Hồi quy Softmax (Multinomial Logistic Regression) tự xây dựng từ đầu bằng NumPy.
    
    Thuật toán:
    1. Đạo hàm ma trận:
       Z = X * W + b             (X: N x D, W: D x C, b: 1 x C)
       P = Softmax(Z)            (P: N x C, P_ij là xác suất mẫu i thuộc lớp j)
    2. Ổn định số học (Numerical Stability):
       Z_stable = Z - max(Z, axis=1, keepdims=True)
       Softmax(Z)_ij = exp(Z_stable_ij) / sum_k(exp(Z_stable_ik))
    3. Hàm mất mát Cross-Entropy Loss:
       Loss = - (1/N) * sum_i sum_c Y_ic * log(P_ic + eps)
    4. Gradient Descent:
       dZ = (P - Y_one_hot) / N
       dW = X.T @ dZ
       db = sum(dZ, axis=0, keepdims=True)
       W = W - lr * dW
       b = b - lr * db
    """
    def __init__(self, learning_rate=0.5, epochs=1500, random_state=42, verbose=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.verbose = verbose
        
        self.W = None                 # Ma trận trọng số (D x C)
        self.b = None                 # Vector bias (1 x C)
        self.loss_history = []        # Lịch sử mất mát trên tập Train
        self.val_loss_history = []    # Lịch sử mất mát trên tập Val (nếu có)
        self.num_classes = None
        self.num_features = None

    def _softmax(self, z):
        """Tính toán hàm Softmax có chống tràn số (Numerical Stability)."""
        # Trừ max mỗi hàng để tránh exp(z) bị tràn số (overflow)
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y, num_classes):
        """Chuyển vector nhãn y thành ma trận one-hot (N x C)."""
        one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float64)
        one_hot[np.arange(y.shape[0]), y] = 1.0
        return one_hot

    def _compute_loss(self, y_one_hot, probs, eps=1e-15):
        """Tính Cross-Entropy Loss có thêm epsilon chống log(0)."""
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        loss = -np.mean(np.sum(y_one_hot * np.log(probs_clipped), axis=1))
        return float(loss)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Huấn luyện mô hình Softmax Regression bằng Gradient Descent.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        N, D = X.shape
        self.num_features = D
        self.num_classes = len(np.unique(y))
        C = self.num_classes
        
        # Khởi tạo trọng số W và bias b bằng 0 (hoặc giá trị ngẫu nhiên nhỏ)
        self.W = np.zeros((D, C), dtype=np.float64)
        self.b = np.zeros((1, C), dtype=np.float64)
        
        y_train_one_hot = self._one_hot_encode(y, C)
        y_val_one_hot = self._one_hot_encode(y_val, C) if y_val is not None else None
        
        self.loss_history = []
        self.val_loss_history = []
        
        if self.verbose:
            print(f" [NumPy Model] Bắt đầu huấn luyện Gradient Descent:")
            print(f"   * Learning rate: {self.learning_rate} | Số epochs: {self.epochs}")
            print(f"   * Số đặc trưng (D): {D} | Số lớp (C): {C} | Kích thước mẫu (N): {N}\n")
            print(f"   {'Epoch':>8} | {'Train Loss':>12} | {'Val Loss':>12} | {'Train Acc':>10}")
            print("   " + "-" * 52)
        
        for epoch in range(1, self.epochs + 1):
            # Forward pass:
            # 1. Tính tổ hợp tuyến tính Z = X * W + b
            Z = np.dot(X, self.W) + self.b
            
            # 2. Tính xác suất bằng hàm Softmax chống tràn
            probs = self._softmax(Z)
            
            # 3. Tính Cross-Entropy Loss
            loss = self._compute_loss(y_train_one_hot, probs)
            self.loss_history.append(loss)
            
            # Nếu có tập validation, tính val_loss
            val_loss_str = "    N/A    "
            if X_val is not None and y_val_one_hot is not None:
                Z_val = np.dot(X_val, self.W) + self.b
                probs_val = self._softmax(Z_val)
                val_loss = self._compute_loss(y_val_one_hot, probs_val)
                self.val_loss_history.append(val_loss)
                val_loss_str = f"{val_loss:12.6f}"
            
            # Backward pass (Tính đạo hàm theo ma trận):
            # dZ = (probs - y_one_hot) / N
            grad_Z = (probs - y_train_one_hot) / N
            
            # dW = X.T @ dZ
            grad_W = np.dot(X.T, grad_Z)
            
            # db = sum(dZ, axis=0, keepdims=True)
            grad_b = np.sum(grad_Z, axis=0, keepdims=True)
            
            # Gradient Descent Weight Update:
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            
            # In tiến trình định kỳ
            if self.verbose and (epoch == 1 or epoch % 200 == 0 or epoch == self.epochs):
                train_acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"   {epoch:>8} | {loss:12.6f} | {val_loss_str} | {train_acc*100:>9.2f}%")
                
        if self.verbose:
            print("   " + "-" * 52)
            print(f" [NumPy Model] Huấn luyện hoàn tất! Loss ban đầu: {self.loss_history[0]:.6f} -> Loss cuối: {self.loss_history[-1]:.6f}\n")
            
        return self

    def predict_proba(self, X):
        """Dự đoán phân phối xác suất cho từng lớp."""
        Z = np.dot(X, self.W) + self.b
        return self._softmax(Z)

    def predict(self, X):
        """Dự đoán nhãn lớp có xác suất cao nhất."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# ======================================================================================
# 4. MỤC 2.5.2: XÂY DỰNG MÔ HÌNH HỒI QUY LOGISTIC BẰNG SCIKIT-LEARN
# ======================================================================================
def train_sklearn_model(X_train, y_train):
    """
    Khởi tạo và huấn luyện mô hình LogisticRegression bằng Scikit-learn.
    Yêu cầu: solver='lbfgs', max_iter=1000, random_state=42.
    """
    print("-" * 80)
    print("[MỤC 2.5.2] XÂY DỰNG MÔ HÌNH HỒI QUY LOGISTIC BẰNG SCIKIT-LEARN")
    print("-" * 80)
    
    # Xử lý tương thích: scikit-learn 1.8+ bỏ tham số deprecated 'multi_class'
    # vì solver='lbfgs' mặc định tự động là multinomial logistic regression
    try:
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        print(" [Scikit-learn] Khởi tạo LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)")
    except TypeError:
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        print(" [Scikit-learn] Khởi tạo LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42) [Tự động xử lý đa lớp multinomial]")
        
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    print(f" [Scikit-learn] Huấn luyện hoàn thành trong: {fit_time:.4f} giây.")
    print(f" [Scikit-learn] Số lần lặp thực tế: {model.n_iter_[0]}")
    return model


# ======================================================================================
# 5. MỤC 2.5.3: ĐÁNH GIÁ & SO SÁNH CHI TIẾT (NUMPY vs SCIKIT-LEARN)
# ======================================================================================
def evaluate_and_compare_models(y_test, y_pred_numpy, y_pred_sklearn, class_names):
    """
    Tính toán và in bảng so sánh chi tiết: Accuracy, Precision, Recall, F1-Score
    giữa mô hình tự cài đặt NumPy và mô hình Scikit-learn.
    """
    print("\n" + "=" * 80)
    print("[MỤC 2.5.3] BẢNG ĐÁNH GIÁ & SO SÁNH CHI TIẾT HIỆU NĂNG TRÊN TẬP TEST")
    print("=" * 80)
    
    # 1. Độ chính xác tổng thể (Overall Accuracy)
    acc_numpy = accuracy_score(y_test, y_pred_numpy)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"\n1. ĐỘ CHÍNH XÁC TỔNG THỂ (OVERALL ACCURACY):")
    print(f"   * Mô hình NumPy (Softmax Regression tự cài đặt) : {acc_numpy * 100:6.2f}% ({acc_numpy:.4f})")
    print(f"   * Mô hình Scikit-learn (L-BFGS Solver)          : {acc_sklearn * 100:6.2f}% ({acc_sklearn:.4f})")
    print(f"   * Chênh lệch (Scikit-learn - NumPy)             : {(acc_sklearn - acc_numpy) * 100:+6.2f}%")
    
    # 2. Chi tiết từng lớp (Per-Class Metrics)
    prec_numpy = precision_score(y_test, y_pred_numpy, average=None)
    rec_numpy = recall_score(y_test, y_pred_numpy, average=None)
    f1_numpy = f1_score(y_test, y_pred_numpy, average=None)
    
    prec_sk = precision_score(y_test, y_pred_sklearn, average=None)
    rec_sk = recall_score(y_test, y_pred_sklearn, average=None)
    f1_sk = f1_score(y_test, y_pred_sklearn, average=None)
    
    print("\n2. SO SÁNH CHI TIẾT THEO TỪNG GIỐNG LÚA (PER-CLASS METRICS):")
    header_fmt = "   | {0:<10} | {1:<18} | {2:<18} | {3:<18} |"
    divider    = "   +" + "-" * 12 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+"
    
    print(divider)
    print(header_fmt.format("Giống lúa", "Precision (NP / SK)", "Recall (NP / SK)", "F1-Score (NP / SK)"))
    print(divider)
    
    for i, name in enumerate(class_names):
        p_str = f"{prec_numpy[i]:.4f} / {prec_sk[i]:.4f}"
        r_str = f"{rec_numpy[i]:.4f} / {rec_sk[i]:.4f}"
        f_str = f"{f1_numpy[i]:.4f} / {f1_sk[i]:.4f}"
        print(header_fmt.format(name, p_str, r_str, f_str))
    print(divider)
    
    # 3. Trung bình Macro và Trung bình Weighted
    p_macro_np = precision_score(y_test, y_pred_numpy, average='macro')
    r_macro_np = recall_score(y_test, y_pred_numpy, average='macro')
    f1_macro_np = f1_score(y_test, y_pred_numpy, average='macro')
    
    p_macro_sk = precision_score(y_test, y_pred_sklearn, average='macro')
    r_macro_sk = recall_score(y_test, y_pred_sklearn, average='macro')
    f1_macro_sk = f1_score(y_test, y_pred_sklearn, average='macro')
    
    p_wt_np = precision_score(y_test, y_pred_numpy, average='weighted')
    r_wt_np = recall_score(y_test, y_pred_numpy, average='weighted')
    f1_wt_np = f1_score(y_test, y_pred_numpy, average='weighted')
    
    p_wt_sk = precision_score(y_test, y_pred_sklearn, average='weighted')
    r_wt_sk = recall_score(y_test, y_pred_sklearn, average='weighted')
    f1_wt_sk = f1_score(y_test, y_pred_sklearn, average='weighted')
    
    print(header_fmt.format("Macro Avg", f"{p_macro_np:.4f} / {p_macro_sk:.4f}", f"{r_macro_np:.4f} / {r_macro_sk:.4f}", f"{f1_macro_np:.4f} / {f1_macro_sk:.4f}"))
    print(header_fmt.format("Weight Avg", f"{p_wt_np:.4f} / {p_wt_sk:.4f}", f"{r_wt_np:.4f} / {r_wt_sk:.4f}", f"{f1_wt_np:.4f} / {f1_wt_sk:.4f}"))
    print(divider)
    
    # 4. In bảng Classification Report chi tiết
    print("\n3. BẢNG CHI TIẾT CLASSIFICATION REPORT (NUMPY):")
    print(classification_report(y_test, y_pred_numpy, target_names=class_names, digits=4))
    
    print("\n4. BẢNG CHI TIẾT CLASSIFICATION REPORT (SCIKIT-LEARN):")
    print(classification_report(y_test, y_pred_sklearn, target_names=class_names, digits=4))


# ======================================================================================
# 6. MỤC 2.5.4: TRỰC QUAN HÓA & XUẤT ĐỒ THỊ KẾT QUẢ
# ======================================================================================
def plot_and_save_visualizations(numpy_model, y_test, y_pred_numpy, y_pred_sklearn, class_names):
    """
    1. Vẽ đồ thị Loss Curve giảm dần theo Epoch của NumPy -> loss_curve_numpy.png
    2. Vẽ 2 Confusion Matrix cạnh nhau để so sánh trực quan -> confusion_matrix_comparison.png
    """
    print("\n" + "-" * 80)
    print("[MỤC 2.5.4] TRỰC QUAN HÓA DỮ LIỆU & LƯU FILE ẢNH")
    print("-" * 80)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ----------------------------------------------------------------------------------
    # 6.1. Vẽ đồ thị hàm mất mát Loss Curve
    # ----------------------------------------------------------------------------------
    loss_curve_path = os.path.join(output_dir, "loss_curve_numpy.png")
    plt.figure(figsize=(9, 6), dpi=300)
    
    epochs_range = range(1, len(numpy_model.loss_history) + 1)
    plt.plot(epochs_range, numpy_model.loss_history, label='Train Cross-Entropy Loss', color='#1f77b4', linewidth=2.2)
    
    if len(numpy_model.val_loss_history) > 0:
        plt.plot(epochs_range, numpy_model.val_loss_history, label='Val Cross-Entropy Loss', color='#ff7f0e', linestyle='--', linewidth=2.0)
        
    plt.title('Đồ Thị Hàm Mất Mát Cross-Entropy (Softmax Regression - NumPy)', fontsize=13, fontweight='bold', pad=14)
    plt.xlabel('Số Epoch (Lần lặp Gradient Descent)', fontsize=11, labelpad=10)
    plt.ylabel('Giá trị Cross-Entropy Loss', fontsize=11, labelpad=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10, loc='upper right')
    
    # Ghi chú giá trị Loss đầu và cuối
    initial_loss = numpy_model.loss_history[0]
    final_loss = numpy_model.loss_history[-1]
    plt.annotate(f'Loss ban đầu: {initial_loss:.4f}', 
                 xy=(1, initial_loss), 
                 xytext=(len(epochs_range)*0.08, initial_loss*0.92),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                 fontsize=9.5, fontweight='semibold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#999999", lw=1))
                 
    plt.annotate(f'Loss hội tụ: {final_loss:.4f}', 
                 xy=(len(epochs_range), final_loss), 
                 xytext=(len(epochs_range)*0.72, final_loss + 0.15),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                 fontsize=9.5, fontweight='semibold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="#2e7d32", lw=1))
                 
    plt.tight_layout()
    plt.savefig(loss_curve_path, bbox_inches='tight')
    plt.close()
    print(f" [ĐỒ THỊ 1] Đã lưu đồ thị đường cong mất mát tại: {loss_curve_path}")
    
    # ----------------------------------------------------------------------------------
    # 6.2. Vẽ 2 Confusion Matrix cạnh nhau
    # ----------------------------------------------------------------------------------
    cm_comparison_path = os.path.join(output_dir, "confusion_matrix_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=300)
    
    cm_np = confusion_matrix(y_test, y_pred_numpy)
    cm_sk = confusion_matrix(y_test, y_pred_sklearn)
    
    # Confusion Matrix: NumPy
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                annot_kws={"size": 11, "weight": "bold"})
    axes[0].set_title(f"Mô hình NumPy (Tự cài đặt Softmax)\nAccuracy: {accuracy_score(y_test, y_pred_numpy)*100:.2f}%", 
                      fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Nhãn thực tế (True Label)", fontsize=11, labelpad=8)
    
    # Confusion Matrix: Scikit-learn
    sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Greens', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                annot_kws={"size": 11, "weight": "bold"})
    axes[1].set_title(f"Mô hình Scikit-learn (L-BFGS Solver)\nAccuracy: {accuracy_score(y_test, y_pred_sklearn)*100:.2f}%", 
                      fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, labelpad=8)
    axes[1].set_ylabel("Nhãn thực tế (True Label)", fontsize=11, labelpad=8)
    
    plt.suptitle("SO SÁNH MA TRẬN NHẦM LẪN (CONFUSION MATRIX) PHÂN LOẠI 4 GIỐNG LÚA", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(cm_comparison_path, bbox_inches='tight')
    plt.close()
    print(f" [ĐỒ THỊ 2] Đã lưu so sánh Confusion Matrix tại: {cm_comparison_path}")


# ======================================================================================
# 7. HÀM MAIN THỰC THI TOÀN BỘ QUY TRÌNH
# ======================================================================================
def main():
    total_start = time.time()
    
    # 1. Đọc dữ liệu
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, feature_cols = load_and_preprocess_data()
    
    # 2. Huấn luyện Mô hình 1: NumPy Softmax Regression
    print("-" * 80)
    print("[MỤC 2.5.1] XÂY DỰNG MÔ HÌNH HỒI QUY LOGISTIC BẰNG NUMPY")
    print("-" * 80)
    numpy_model = SoftmaxRegressionNumPy(
        learning_rate=0.5,
        epochs=1500,
        random_state=42,
        verbose=True
    )
    t0 = time.time()
    numpy_model.fit(X_train, y_train, X_val, y_val)
    numpy_fit_time = time.time() - t0
    print(f" [NumPy Model] Thời gian huấn luyện: {numpy_fit_time:.4f} giây")
    
    # 3. Huấn luyện Mô hình 2: Scikit-learn Logistic Regression
    sklearn_model = train_sklearn_model(X_train, y_train)
    
    # 4. Dự đoán trên tập kiểm thử (Test Set)
    print("\n" + "-" * 80)
    print("TIẾN HÀNH DỰ ĐOÁN TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("-" * 80)
    y_pred_numpy = numpy_model.predict(X_test)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    # 5. Đánh giá và so sánh chi tiết
    evaluate_and_compare_models(y_test, y_pred_numpy, y_pred_sklearn, class_names)
    
    # 6. Trực quan hóa và lưu ảnh
    plot_and_save_visualizations(numpy_model, y_test, y_pred_numpy, y_pred_sklearn, class_names)
    
    print("\n" + "=" * 80)
    print(f"HOÀN THÀNH TOÀN BỘ QUY TRÌNH THÀNH CÔNG! TỔNG THỜI GIAN: {time.time() - total_start:.2f} GIÂY")
    print("=" * 80)


if __name__ == "__main__":
    main()
