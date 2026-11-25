"""
========================================================================================
ỨNG DỤNG NHẬN DẠNG GIỐNG LÚA TỪ 7 ĐẶC TRƯNG HÌNH THÁI HẠT GẠO
SỬ DỤNG THUẬT TOÁN HỒI QUY LOGISTIC (LOGISTIC REGRESSION) & GIAO DIỆN TKINTER
========================================================================================
- Dữ liệu: Rice_4Classes_Scaled_Splitted.xlsx (4 giống lúa: Cammeo, Gonen, Jasmine, Osmancik)
- 7 Đặc trưng hình học: Area, Perimeter, Major_Axis_Length, Minor_Axis_Length,
                       Eccentricity, Convex_Area, Extent
- Giao diện: Tkinter GUI hiện đại, chuyên nghiệp, hỗ trợ nạp mẫu kiểm thử nhanh
========================================================================================
"""

import sys
import os
import subprocess
import time

# Đảm bảo in UTF-8 không lỗi trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ======================================================================================
# 1. KIỂM TRA VÀ CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ======================================================================================
def check_and_install_dependencies():
    """Tự động kiểm tra và cài đặt các thư viện cần thiết nếu chưa có."""
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl"
    }
    
    print("-" * 80)
    print("[BƯỚC 1] KIỂM TRA MÔI TRƯỜNG VÀ THƯ VIỆN CẦN THIẾT")
    print("-" * 80)
    
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
            print(f" [OK] Thư viện '{pip_name}' đã được cài đặt.")
        except ImportError:
            print(f" [..] Đang cài đặt thư viện thiếu: '{pip_name}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f" [OK] Cài đặt thành công '{pip_name}'.")

    # Kiểm tra riêng Tkinter
    try:
        import tkinter
        print(" [OK] Giao diện Tkinter sẵn sàng hoạt động.")
    except ImportError:
        print(" [CẢNH BÁO] Tkinter chưa được tích hợp trong Python này!")

check_and_install_dependencies()

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


# ======================================================================================
# 2. ĐỌC DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH HỒI QUY LOGISTIC
# ======================================================================================
def load_data_and_train_model():
    """Đọc dữ liệu từ file Excel, huấn luyện mô hình LogisticRegression và đánh giá."""
    print("\n" + "-" * 80)
    print("[BƯỚC 2] ĐỌC DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH")
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
        raise FileNotFoundError("Không tìm thấy file Rice_4Classes_Scaled_Splitted.xlsx trong hệ thống!")
        
    print(f" Nạp dữ liệu từ: {dataset_path}")
    excel_file = pd.ExcelFile(dataset_path)
    sheet_names = excel_file.sheet_names
    print(f" Danh sách sheet: {sheet_names}")
    
    train_sheet = next((s for s in sheet_names if 'train' in s.lower()), None)
    test_sheet = next((s for s in sheet_names if 'test' in s.lower()), None)
    
    if train_sheet and test_sheet:
        train_df = excel_file.parse(train_sheet)
        test_df = excel_file.parse(test_sheet)
    else:
        print(" Không tìm thấy sheet Train/Test riêng biệt, tự động phân chia Stratified 80/20...")
        df = excel_file.parse(sheet_names[0])
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
        
    label_col = 'Class' if 'Class' in train_df.columns else train_df.columns[-1]
    feature_cols = [c for c in train_df.columns if c != label_col][:7]
    
    # Mã hóa nhãn (Label Encoding)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_test = le.transform(test_df[label_col])
    
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    class_names = list(le.classes_)
    
    print(f" 7 Đặc trưng hình học sử dụng: {feature_cols}")
    print(f" 4 Giống lúa phân loại: {class_names}")
    print(f" Kích thước tập Train: {X_train.shape} | Kích thước tập Test: {X_test.shape}")
    
    # Huấn luyện mô hình Logistic Regression đa lớp
    try:
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
    except TypeError:
        model = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
    start_t = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_t
    
    # Đánh giá trên tập Test
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 80)
    print(f" KẾT QUẢ HUẤN LUYỆN TRÊN TẬP TEST:")
    print(f" * Thời gian huấn luyện: {train_time:.4f} giây")
    print(f" * Độ chính xác (Accuracy): {test_accuracy * 100:.2f}% ({test_accuracy:.4f})")
    print("=" * 80)
    print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    return model, le, feature_cols, class_names, test_accuracy, test_df


# ======================================================================================
# 3. GIAO DIỆN ĐỒ HỌA DỰ ĐOÁN BẰNG TKINTER
# ======================================================================================
class RiceClassificationApp:
    def __init__(self, root, model, label_encoder, feature_cols, class_names, test_accuracy, test_df):
        self.root = root
        self.model = model
        self.le = label_encoder
        self.feature_cols = feature_cols
        self.class_names = class_names
        self.test_accuracy = test_accuracy
        self.test_df = test_df
        
        # Mô tả tiếng Việt cho từng đặc trưng
        self.feature_descriptions = {
            'Area': '1. Diện tích hạt (Area)',
            'Perimeter': '2. Chu vi hạt (Perimeter)',
            'Major_Axis_Length': '3. Chiều dài trục chính (Major Axis)',
            'Minor_Axis_Length': '4. Chiều dài trục phụ (Minor Axis)',
            'Eccentricity': '5. Độ lệch tâm / Độ tròn (Eccentricity)',
            'Convex_Area': '6. Diện tích bao lồi (Convex Area)',
            'Extent': '7. Tỷ lệ bao phủ hạt (Extent)'
        }
        
        # Dữ liệu mẫu chuẩn hóa đặc trưng cho từng giống lúa (lấy từ tập Test)
        self.sample_profiles = {
            "Cammeo": [1.928965, 1.978245, 2.039589, 1.522307, -0.270489, 1.893502, -0.007056],
            "Gonen": [0.133107, 0.112201, 0.007224, 0.276678, -0.261074, 0.166460, -0.836893],
            "Jasmine": [-1.220763, -1.549641, -1.497292, -1.135462, 0.494464, -1.238052, -0.450278],
            "Osmancik": [1.499596, 1.377800, 1.067458, 1.598084, -1.123189, 1.492963, 0.139958]
        }
        
        # Thông tin đặc điểm sinh học của từng giống
        self.rice_info = {
            "Cammeo": "🌾 Cammeo: Giống lúa hạt dài và to, diện tích và chu vi hạt lớn vượt trội, độ nở cao khi nấu.",
            "Gonen": "🌾 Gonen: Giống lúa hạt kích thước trung bình, hình dạng cân đối, độ nở và độ dẻo vừa phải.",
            "Jasmine": "🌾 Jasmine: Giống lúa thon dài đặc trưng (diện tích và trục phụ nhỏ nhất), độ lệch tâm cao, hương thơm tự nhiên.",
            "Osmancik": "🌾 Osmancik: Giống lúa hạt tròn bầu dục, độ lệch tâm thấp (rất tròn), hàm lượng tinh bột cao, phổ biến tại Thổ Nhĩ Kỳ."
        }
        
        # Màu sắc đại diện cho từng lớp
        self.class_colors = {
            "Cammeo": "#D97706",    # Hổ phách / Vàng cam
            "Gonen": "#2563EB",     # Xanh dương
            "Jasmine": "#059669",   # Xanh ngọc lục bảo
            "Osmancik": "#7C3AED"   # Tím violet
        }
        
        self.entries = {}
        self.prob_labels = {}
        self.prob_bars = {}
        
        self._setup_window()
        self._build_ui()
        
        # Tự động nạp mẫu đầu tiên (Jasmine) để người dùng thấy kết quả ngay
        self.load_sample("Jasmine")
        self.predict()

    def _setup_window(self):
        self.root.title(f"NHẬN DẠNG GIỐNG LÚA - LOGISTIC REGRESSION (Độ chính xác: {self.test_accuracy*100:.2f}%)")
        self.root.geometry("1060x740")
        self.root.minsize(980, 680)
        self.root.configure(bg="#F1F5F9")
        
        # Canh giữa màn hình
        self.root.update_idletasks()
        w = 1060
        h = 740
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2) - 30
        self.root.geometry(f'{w}x{h}+{x}+{y}')

    def _build_ui(self):
        # 1. HEADER BANNER
        header_frame = tk.Frame(self.root, bg="#0F172A", pady=14, padx=20)
        header_frame.pack(fill="x", side="top")
        
        title_label = tk.Label(
            header_frame,
            text="🌾 HỆ THỐNG NHẬN DẠNG 4 GIỐNG LÚA TỪ HÌNH THÁI HẠT GẠO",
            font=("Segoe UI", 16, "bold"),
            fg="#F8FAFC",
            bg="#0F172A"
        )
        title_label.pack(anchor="w")
        
        subtitle_label = tk.Label(
            header_frame,
            text=f"Thuật toán: Hồi quy Logistic (Multinomial L-BFGS)  |  Độ chính xác trên tập Test: {self.test_accuracy*100:.2f}%  |  7 Đặc trưng hình học chuẩn hóa",
            font=("Segoe UI", 10),
            fg="#94A3B8",
            bg="#0F172A"
        )
        subtitle_label.pack(anchor="w", pady=(3, 0))

        # 2. MAIN CONTAINER (2 Cột)
        main_container = tk.Frame(self.root, bg="#F1F5F9", padx=16, pady=14)
        main_container.pack(fill="both", expand=True)
        main_container.columnconfigure(0, weight=5)
        main_container.columnconfigure(1, weight=5)
        main_container.rowconfigure(0, weight=1)

        # ------------------------------------------------------------------------------
        # CỘT TRÁI: KHUNG NHẬP LIỆU ĐẶC TRƯNG HÌNH HỌC
        # ------------------------------------------------------------------------------
        left_frame = tk.LabelFrame(
            main_container,
            text=" ⚙ 7 ĐẶC TRƯNG HÌNH HỌC HẠT GẠO (INPUT) ",
            font=("Segoe UI", 11, "bold"),
            fg="#1E293B",
            bg="#FFFFFF",
            relief="groove",
            bd=2,
            padx=14,
            pady=12
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Thanh nạp mẫu nhanh (Quick Sample Loader)
        sample_bar = tk.LabelFrame(
            left_frame,
            text=" Thử nghiệm nhanh với mẫu từ tập Test: ",
            font=("Segoe UI", 9, "italic"),
            fg="#475569",
            bg="#F8FAFC",
            padx=6,
            pady=6
        )
        sample_bar.pack(fill="x", pady=(0, 10))

        btn_row = tk.Frame(sample_bar, bg="#F8FAFC")
        btn_row.pack(fill="x")
        
        for variety in ["Cammeo", "Gonen", "Jasmine", "Osmancik"]:
            btn = tk.Button(
                btn_row,
                text=variety,
                font=("Segoe UI", 8, "bold"),
                bg="#E2E8F0",
                fg="#1E293B",
                activebackground=self.class_colors[variety],
                activeforeground="#FFFFFF",
                cursor="hand2",
                relief="ridge",
                padx=6,
                pady=3,
                command=lambda v=variety: self.load_sample(v)
            )
            btn.pack(side="left", padx=3, expand=True, fill="x")

        btn_random = tk.Button(
            btn_row,
            text="🎲 Mẫu Ngẫu Nhiên",
            font=("Segoe UI", 8, "bold"),
            bg="#CBD5E1",
            fg="#0F172A",
            cursor="hand2",
            relief="ridge",
            padx=6,
            pady=3,
            command=self.load_random_sample
        )
        btn_random.pack(side="left", padx=3, expand=True, fill="x")

        # 7 Ô nhập liệu tương ứng 7 đặc trưng hình học
        inputs_container = tk.Frame(left_frame, bg="#FFFFFF")
        inputs_container.pack(fill="both", expand=True)

        for idx, feat in enumerate(self.feature_cols):
            row_frame = tk.Frame(inputs_container, bg="#FFFFFF", pady=3)
            row_frame.pack(fill="x")
            
            lbl_text = self.feature_descriptions.get(feat, feat)
            lbl = tk.Label(
                row_frame,
                text=lbl_text,
                font=("Segoe UI", 9, "bold"),
                fg="#334155",
                bg="#FFFFFF",
                width=34,
                anchor="w"
            )
            lbl.pack(side="left")
            
            entry = tk.Entry(
                row_frame,
                font=("Segoe UI", 10),
                bg="#F8FAFC",
                fg="#0F172A",
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightcolor="#2563EB",
                width=16
            )
            entry.pack(side="right", padx=(5, 0), ipady=3)
            self.entries[feat] = entry

        # Nút bấm Dự đoán & Xóa
        btn_action_frame = tk.Frame(left_frame, bg="#FFFFFF", pady=10)
        btn_action_frame.pack(fill="x", side="bottom")

        predict_btn = tk.Button(
            btn_action_frame,
            text="🔍 DỰ ĐOÁN GIỐNG LÚA",
            font=("Segoe UI", 11, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            cursor="hand2",
            relief="flat",
            pady=8,
            command=self.predict
        )
        predict_btn.pack(fill="x", pady=(0, 6))

        clear_btn = tk.Button(
            btn_action_frame,
            text="🔄 Xóa Trắng / Nhập Mới",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#475569",
            cursor="hand2",
            relief="solid",
            bd=1,
            pady=4,
            command=self.clear_fields
        )
        clear_btn.pack(fill="x")

        # ------------------------------------------------------------------------------
        # CỘT PHẢI: KHUNG HIỂN THỊ KẾT QUẢ DỰ ĐOÁN
        # ------------------------------------------------------------------------------
        right_frame = tk.LabelFrame(
            main_container,
            text=" 📊 KẾT QUẢ NHẬN DẠNG & PHÂN BỐ XÁC SUẤT (OUTPUT) ",
            font=("Segoe UI", 11, "bold"),
            fg="#1E293B",
            bg="#FFFFFF",
            relief="groove",
            bd=2,
            padx=16,
            pady=14
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Card Kết quả chính (Dự đoán nổi bật)
        self.result_card = tk.Frame(right_frame, bg="#ECFDF5", bd=2, relief="solid")
        self.result_card.configure(highlightbackground="#10B981", highlightthickness=2)
        self.result_card.pack(fill="x", pady=(0, 14), ipady=8)

        self.res_tag_label = tk.Label(
            self.result_card,
            text="GIỐNG LÚA DỰ ĐOÁN ĐƯỢC XÁC ĐỊNH LÀ:",
            font=("Segoe UI", 9, "bold"),
            fg="#047857",
            bg="#ECFDF5"
        )
        self.res_tag_label.pack(pady=(6, 0))

        self.pred_variety_label = tk.Label(
            self.result_card,
            text="JASMINE",
            font=("Segoe UI", 22, "bold"),
            fg="#059669",
            bg="#ECFDF5"
        )
        self.pred_variety_label.pack()

        self.confidence_label = tk.Label(
            self.result_card,
            text="Độ tin cậy (Xác suất cao nhất): 99.91%",
            font=("Segoe UI", 11, "bold"),
            fg="#065F46",
            bg="#ECFDF5"
        )
        self.confidence_label.pack(pady=(2, 4))

        self.rice_img_label = tk.Label(self.result_card, bg="#ECFDF5")
        self.rice_img_label.pack(pady=(2, 6))

        # Phân bố xác suất 4 giống lúa
        prob_header = tk.Label(
            right_frame,
            text="PHÂN BỐ CHI TIẾT XÁC SUẤT CỦA 4 GIỐNG LÚA:",
            font=("Segoe UI", 10, "bold"),
            fg="#334155",
            bg="#FFFFFF"
        )
        prob_header.pack(anchor="w", pady=(4, 8))

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TProgressbar", thickness=16, troughcolor="#F1F5F9")

        for variety in self.class_names:
            card = tk.Frame(right_frame, bg="#F8FAFC", relief="ridge", bd=1, padx=8, pady=6)
            card.pack(fill="x", pady=3)
            
            top_line = tk.Frame(card, bg="#F8FAFC")
            top_line.pack(fill="x")
            
            # Badge tên giống
            lbl_name = tk.Label(
                top_line,
                text=f"🌾 {variety}",
                font=("Segoe UI", 10, "bold"),
                fg=self.class_colors.get(variety, "#0F172A"),
                bg="#F8FAFC"
            )
            lbl_name.pack(side="left")
            
            # Label % xác suất
            lbl_val = tk.Label(
                top_line,
                text="0.00%",
                font=("Segoe UI", 10, "bold"),
                fg="#1E293B",
                bg="#F8FAFC"
            )
            lbl_val.pack(side="right")
            self.prob_labels[variety] = lbl_val
            
            # Thanh Progress Bar
            pbar = ttk.Progressbar(card, style="TProgressbar", orient="horizontal", mode="determinate", maximum=100)
            pbar.pack(fill="x", pady=(3, 0))
            self.prob_bars[variety] = pbar

        # Khung thông tin mô tả giống lúa dự đoán
        self.info_box = tk.LabelFrame(
            right_frame,
            text=" ℹ Đặc điểm giống lúa nhận diện: ",
            font=("Segoe UI", 9, "bold"),
            fg="#475569",
            bg="#FFFFFF",
            padx=10,
            pady=8
        )
        self.info_box.pack(fill="both", expand=True, pady=(12, 0))

        self.info_label = tk.Label(
            self.info_box,
            text=self.rice_info.get("Jasmine", ""),
            font=("Segoe UI", 9),
            fg="#334155",
            bg="#FFFFFF",
            wraplength=440,
            justify="left"
        )
        self.info_label.pack(anchor="nw", fill="both", expand=True)

        # 3. FOOTER STATUS BAR
        status_frame = tk.Frame(self.root, bg="#E2E8F0", padx=14, pady=4)
        status_frame.pack(fill="x", side="bottom")

        status_lbl = tk.Label(
            status_frame,
            text="Trạng thái: Mô hình sẵn sàng  |  Tập huấn luyện: 15,396 mẫu  |  Tập kiểm thử: 3,300 mẫu  |  Nhãn: 4 giống lúa chuẩn hóa",
            font=("Segoe UI", 8),
            fg="#475569",
            bg="#E2E8F0"
        )
        status_lbl.pack(side="left")

    def load_sample(self, variety_name):
        """Nạp dữ liệu mẫu của giống lúa tương ứng vào 7 ô nhập liệu."""
        if variety_name in self.sample_profiles:
            vals = self.sample_profiles[variety_name]
            for feat, val in zip(self.feature_cols, vals):
                self.entries[feat].delete(0, tk.END)
                self.entries[feat].insert(0, f"{val:.6f}")
            self.predict()

    def load_random_sample(self):
        """Lấy ngẫu nhiên 1 mẫu từ tập kiểm thử test_df."""
        if self.test_df is not None and not self.test_df.empty:
            random_row = self.test_df.sample(n=1, random_state=int(time.time() * 1000) % 10000).iloc[0]
            true_label = random_row['Class']
            for feat in self.feature_cols:
                val = float(random_row[feat])
                self.entries[feat].delete(0, tk.END)
                self.entries[feat].insert(0, f"{val:.6f}")
            self.predict()
            # Báo tên nhãn gốc của mẫu ngẫu nhiên
            messagebox.showinfo("Mẫu ngẫu nhiên", f"Đã nạp 1 mẫu ngẫu nhiên từ tập Test!\nGiống lúa thực tế của mẫu này là: '{true_label}'")

    def clear_fields(self):
        """Xóa trắng toàn bộ các ô nhập liệu."""
        for feat in self.feature_cols:
            self.entries[feat].delete(0, tk.END)
        self.pred_variety_label.config(text="CHƯA CÓ KẾT QUẢ", fg="#64748B")
        self.confidence_label.config(text="Vui lòng nhập số liệu và bấm Dự đoán", fg="#64748B")
        for variety in self.class_names:
            self.prob_labels[variety].config(text="0.00%")
            self.prob_bars[variety]['value'] = 0
        self.info_label.config(text="Hãy nhập 7 đặc trưng hình học hoặc chọn 1 mẫu thử nghiệm để xem phân tích.")

    def predict(self):
        """Đọc 7 giá trị từ giao diện, kiểm tra tính hợp lệ và thực hiện dự đoán."""
        input_values = []
        for feat in self.feature_cols:
            val_str = self.entries[feat].get().strip()
            if not val_str:
                messagebox.showwarning(
                    "Thiếu dữ liệu nhập",
                    f"Vui lòng không để trống ô: '{self.feature_descriptions[feat]}'."
                )
                self.entries[feat].focus_set()
                return
            try:
                val = float(val_str)
                input_values.append(val)
            except ValueError:
                messagebox.showerror(
                    "Định dạng số không hợp lệ",
                    f"Giá trị tại ô '{self.feature_descriptions[feat]}' phải là số thực (ví dụ: 1.928 hoặc -0.27)!\nGiá trị vừa nhập: '{val_str}'"
                )
                self.entries[feat].focus_set()
                return

        # Kiểm tra hình thái sai lệch / toàn bộ số 0
        if all(v == 0.0 for v in input_values):
            messagebox.showwarning(
                "Hình thái không hợp lệ",
                "Tất cả các thông số hình thái đều bằng 0!\nVui lòng nhập thông số đo đạc thực tế của hạt gạo hoặc chọn 'Nạp mẫu'."
            )
            return

        # Kiểm tra bất biến hình thái cơ bản (chỉ áp dụng nếu người dùng nhập ở thang đo pixel thực tế > 100)
        # Nếu nhập giá trị chuẩn hóa Z-score từ tập Test (-3 đến +3) thì không so sánh trực tiếp vì độ lệch chuẩn các trục là khác nhau
        val_map = dict(zip(self.feature_cols, input_values))
        errs = []
        if val_map.get('Area', 0) > 100:
            if 'Minor_Axis_Length' in val_map and 'Major_Axis_Length' in val_map:
                if val_map['Minor_Axis_Length'] > val_map['Major_Axis_Length'] and val_map['Major_Axis_Length'] > 0:
                    errs.append("Trục phụ (chiều rộng) không thể lớn hơn trục chính (chiều dài).")
            if 'Area' in val_map and 'Convex_Area' in val_map:
                if val_map['Area'] > val_map['Convex_Area'] and val_map['Convex_Area'] > 0:
                    errs.append("Diện tích hạt không thể lớn hơn diện tích bao lồi.")

        if errs:
            messagebox.showwarning("Sai lệch hình học hạt gạo", "\n".join(errs))
            return

        # Tự động nhận diện định dạng:
        # - Nếu Area > 100: Người dùng nhập kích thước pixel thực tế -> Chuẩn hóa Z-score ngầm
        # - Nếu Area <= 100: Người dùng nhập giá trị Z-score từ tập Test -> Dùng trực tiếp
        if val_map.get('Area', 0) > 100:
            mu = {'Area': 8016.862302, 'Perimeter': 369.511365, 'Major_Axis_Length': 158.141652,
                  'Minor_Axis_Length': 64.428029, 'Eccentricity': 0.910401, 'Convex_Area': 8223.150883, 'Extent': 0.624796}
            sigma = {'Area': 2615.463556, 'Perimeter': 49.514917, 'Major_Axis_Length': 19.445475,
                     'Minor_Axis_Length': 13.775489, 'Eccentricity': 0.031060, 'Convex_Area': 2666.408241, 'Extent': 0.101458}
            scaled_vals = [(val_map[f] - mu[f]) / sigma[f] for f in self.feature_cols]
            X_input = np.array([scaled_vals], dtype=np.float64)
        else:
            X_input = np.array([input_values], dtype=np.float64)

        pred_idx = self.model.predict(X_input)[0]
        probs = self.model.predict_proba(X_input)[0]

        pred_class_name = self.class_names[pred_idx]
        max_prob = probs[pred_idx] * 100

        # Cập nhật Card kết quả
        class_color = self.class_colors.get(pred_class_name, "#2563EB")
        self.pred_variety_label.config(text=pred_class_name.upper(), fg=class_color)
        self.confidence_label.config(
            text=f"Độ tin cậy (Xác suất cao nhất): {max_prob:.2f}%",
            fg=class_color
        )
        self.result_card.configure(highlightbackground=class_color)

        # Cập nhật ảnh minh họa thực tế hạt gạo
        img_candidates = [
            os.path.join(os.path.dirname(__file__), "assets", "rice_images", f"{pred_class_name}.jpg"),
            os.path.join(os.path.dirname(__file__), "Ảnh", f"{pred_class_name}.jpg"),
            os.path.join(os.path.dirname(__file__), f"{pred_class_name}.jpg")
        ]
        img_found = next((p for p in img_candidates if os.path.exists(p)), None)
        if img_found:
            try:
                pil_img = Image.open(img_found).resize((220, 125), Image.Resampling.LANCZOS)
                self.tk_photo = ImageTk.PhotoImage(pil_img)
                self.rice_img_label.config(image=self.tk_photo)
            except Exception:
                pass

        # Cập nhật phân bố xác suất cho cả 4 lớp
        for idx, variety in enumerate(self.class_names):
            p = float(probs[idx] * 100)
            self.prob_labels[variety].config(text=f"{p:6.2f}%")
            self.prob_bars[variety]['value'] = p

        # Cập nhật thông tin chi tiết về giống lúa
        desc = self.rice_info.get(pred_class_name, "Không có thông tin mô tả bổ sung.")
        self.info_label.config(text=desc)


# ======================================================================================
# 4. HÀM MAIN KHỞI CHẠY ỨNG DỤNG
# ======================================================================================
def main():
    # 1. Huấn luyện mô hình
    model, le, feature_cols, class_names, test_accuracy, test_df = load_data_and_train_model()
    
    # 2. Khởi tạo cửa sổ Tkinter
    print("\n" + "-" * 80)
    print("[BƯỚC 3] KHỞI ĐỘNG GIAO DIỆN ĐỒ HỌA TKINTER...")
    print("-" * 80)
    print(" Đang hiển thị cửa sổ ứng dụng 'NHẬN DẠNG GIỐNG LÚA' lên màn hình...")
    
    root = tk.Tk()
    app = RiceClassificationApp(
        root=root,
        model=model,
        label_encoder=le,
        feature_cols=feature_cols,
        class_names=class_names,
        test_accuracy=test_accuracy,
        test_df=test_df
    )
    
    root.mainloop()
    print(" Ứng dụng giao diện đã được đóng thành công.")


if __name__ == "__main__":
    main()

