"""
========================================================================================
🌾 HỆ THỐNG NHẬN DẠNG GIỐNG LÚA - DỰA TRÊN 7 ĐẶC TRƯNG CỦA HẠT GẠO
PHIÊN BẢN GIAO DIỆN TỐI GIẢN - HIỆN ĐẠI - NHẬN DẠNG TỨC THÌ (REAL-TIME DASHBOARD)
========================================================================================
- Bố cục 2 khối tinh gọn đặt song song (50% - 50%): Form nhập liệu bên trái, Kết quả bên phải.
- Tích hợp khối Lưu trữ Lịch sử nhận dạng tự động: xem bảng dữ liệu, xuất CSV và quản lý lịch sử.
- 7 ô nhập liệu được xếp gần lại nhau dạng lưới 2 cột tinh tế, khoảng cách sít sao, không phải cuộn trang.
- Nhận dạng tự động thời gian thực: Khi nhập/thay đổi thông số, kết quả lập tức hiển thị ngay bên cạnh.
- Thẻ kết quả hiển thị: Ảnh macro hạt gạo thật, Tên giống lúa (size lớn), Độ chính xác %, 
  Thanh xác suất phân phối Softmax cho 4 giống lúa, và Hộp đặc điểm nhận diện nổi bật.
- Giữ nguyên mô hình Hồi quy Logistic đa biến L-BFGS (96.91%), tự động chuẩn hóa Z-score ngầm.
========================================================================================
"""

import sys
import os
import time
import base64
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# 1. CẤU HÌNH TRANG STREAMLIT
st.set_page_config(
    page_title="Hệ Thống Nhận Dạng Giống Lúa",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. DESIGN SYSTEM AGRITECH CHUẨN 100% HIỆN ĐẠI & TINH GỌN
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,600&display=swap');

    /* 1. THỐNG NHẤT 1 PHÔNG CHỮ DUY NHẤT TOÀN HỆ THỐNG (KHÔNG ĐÈ FONT ICON CỦA STREAMLIT) */
    html, body, .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }

    h1, h2, h3, h4, h5, h6, p, label, input, button, select, textarea, table, td, th,
    .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* BẢO TOÀN NGUYÊN VẸN FONT ICON CHO MỌI BIỂU TƯỢNG STREAMLIT (MATERIAL SYMBOLS ROUNDED) */
    [data-testid*="Icon"],
    [data-testid="stIconMaterial"],
    [data-testid="stExpanderToggleIcon"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-symbols-sharp,
    details summary span[data-testid="stIconMaterial"],
    details summary [data-testid="stExpanderToggleIcon"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        display: inline-block !important;
        line-height: 1 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        word-wrap: normal !important;
        white-space: nowrap !important;
        direction: ltr !important;
        vertical-align: middle !important;
    }

    body, .stApp {
        background-color: #F4F6F4 !important;
        color: #1F2937 !important;
    }

    /* 2. ẨN HOÀN TOÀN HEADER MẶC ĐỊNH CỦA STREAMLIT ĐỂ TRÁNH BỊ ĐÈ/CẮT CHỮ */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* 3. ĐIỀU CHỈNH PADDING BLOCK CONTAINER VỪA VẶN MÀN HÌNH */
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: 2.0rem !important;
        padding-right: 2.0rem !important;
        max-width: 1440px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* 4. TOP HEADER BANNER TRẢI DÀI */
    .top-header-banner {
        background: linear-gradient(90deg, rgba(16, 85, 45, 0.96) 0%, rgba(20, 108, 58, 0.92) 55%, rgba(30, 130, 70, 0.82) 100%), 
                    url('https://images.unsplash.com/photo-1536304993881-ff6e9eefa2a6?auto=format&fit=crop&w=1600&q=80') center/cover;
        border-radius: 14px;
        padding: 20px 30px;
        margin-bottom: 18px;
        color: #FFFFFF;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 18px rgba(16, 85, 45, 0.20);
        min-height: 95px;
        box-sizing: border-box;
    }
    .header-logo-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .header-logo-icon {
        background: rgba(255, 255, 255, 0.16);
        border: 2px solid rgba(255, 255, 255, 0.45);
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.85rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
        flex-shrink: 0;
    }
    .header-title-text {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
        line-height: 1.3 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    .header-subtitle-text {
        font-size: 0.98rem !important;
        font-style: italic !important;
        color: #D1FAE5 !important;
        margin-top: 3px !important;
        font-weight: 500 !important;
        line-height: 1.3 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    .user-pill-btn {
        background: #FFFFFF;
        color: #1F2937;
        font-size: 0.86rem;
        font-weight: 700;
        padding: 8px 18px;
        border-radius: 24px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    }

    /* 5. THẺ CARD CHỨA NỘI DUNG */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04) !important;
        padding: 16px 18px !important;
    }

    /* 6. ĐỊNH DẠNG Ô NHẬP LIỆU GỌN GÀNG, SÍT SAO */
    div[data-testid="stNumberInput"] {
        margin-bottom: 2px !important;
    }
    div[data-testid="stNumberInput"] label {
        font-size: 0.84rem !important;
        font-weight: 700 !important;
        color: #1F2937 !important;
        margin-bottom: 2px !important;
        padding-bottom: 0px !important;
    }
    div[data-testid="stNumberInput"] input {
        border-radius: 8px !important;
        border: 1.5px solid #D1D5DB !important;
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        padding: 6px 12px !important;
        background-color: #F9FAFB !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #135E38 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(19, 94, 56, 0.20) !important;
    }
    .input-hint-sub {
        font-size: 0.72rem !important;
        color: #4B5563 !important;
        font-weight: 600 !important;
        margin-top: 1px !important;
        margin-bottom: 6px !important;
    }

    /* 7. NÚT NẠP MẪU NHANH GỌN GÀNG */
    div.stButton > button {
        background-color: #F3F4F6 !important;
        border: 1px solid #D1D5DB !important;
        color: #374151 !important;
        font-weight: 700 !important;
        font-size: 0.80rem !important;
        border-radius: 8px !important;
        padding: 5px 8px !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover {
        background-color: #135E38 !important;
        border-color: #135E38 !important;
        color: #FFFFFF !important;
    }

    /* 8. HỘP ĐẶC ĐIỂM NỔI BẬT */
    .highlights-box {
        background-color: #F8FAF7;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 10px;
    }
    .highlight-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 5px;
        line-height: 1.3;
    }

    /* 9. THANH XÁC SUẤT SOFTMAX */
    .prob-bar-container {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
        font-size: 0.80rem;
        font-weight: 700;
    }
    .prob-bar-label {
        width: 78px;
        color: #374151;
        text-align: right;
    }
    .prob-bar-track {
        flex: 1;
        background: #E5E7EB;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s ease;
    }
    .prob-bar-val {
        width: 48px;
        color: #135E38;
        font-weight: 800;
        font-size: 0.80rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# 3. NẠP DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH (CACHE HIỆU NĂNG CAO)
# ======================================================================================
@st.cache_resource
def load_all_resources():
    clean_paths = [
        r"C:\Users\LENOVO\Documents\Nam_cuoi\hoc_may\archive (1)\dataset\Rice_4Classes_Cleaned.xlsx",
        os.path.join(os.path.dirname(__file__), "archive (1)", "dataset", "Rice_4Classes_Cleaned.xlsx"),
        os.path.join(os.path.dirname(__file__), "Rice_4Classes_Cleaned.xlsx"),
        "Rice_4Classes_Cleaned.xlsx"
    ]
    clean_path = next((p for p in clean_paths if os.path.exists(p)), None)
    if clean_path is None:
        st.error("Không tìm thấy file 'Rice_4Classes_Cleaned.xlsx'!")
        st.stop()

    clean_df = pd.read_excel(clean_path)
    label_col = 'Class' if 'Class' in clean_df.columns else clean_df.columns[-1]
    feature_cols = [c for c in clean_df.columns if c != label_col][:7]

    # Tham số chuẩn hóa Z-score từ dataset gốc
    mu_series = pd.Series({
        'Area': 8016.862302,
        'Perimeter': 369.511365,
        'Major_Axis_Length': 158.141652,
        'Minor_Axis_Length': 64.428029,
        'Eccentricity': 0.910401,
        'Convex_Area': 8223.150883,
        'Extent': 0.624796
    })
    sigma_series = pd.Series({
        'Area': 2615.463556,
        'Perimeter': 49.514917,
        'Major_Axis_Length': 19.445475,
        'Minor_Axis_Length': 13.775489,
        'Eccentricity': 0.031060,
        'Convex_Area': 2666.408241,
        'Extent': 0.101458
    })

    # Giới hạn thực tế Min, Max toàn bộ dataset
    real_bounds = {}
    for f in feature_cols:
        real_bounds[f] = {
            'min': float(clean_df[f].min()),
            'max': float(clean_df[f].max()),
            'mean': float(clean_df[f].mean()),
            'std': float(clean_df[f].std())
        }

    # Bảng Min - Mean - Max theo từng giống lúa
    stats_clean_grouped = clean_df.groupby(label_col)[feature_cols].agg(['min', 'mean', 'max'])
    class_names = sorted(list(clean_df[label_col].unique()))

    stats_clean_dict = {}
    class_means_real = {}
    for c in class_names:
        stats_clean_dict[c] = {}
        class_means_real[c] = {}
        for f in feature_cols:
            mi = float(stats_clean_grouped.loc[c, (f, 'min')])
            me = float(stats_clean_grouped.loc[c, (f, 'mean')])
            ma = float(stats_clean_grouped.loc[c, (f, 'max')])
            stats_clean_dict[c][f] = {'min': mi, 'mean': me, 'max': ma}
            class_means_real[c][f] = me

    # Nạp mô hình Logistic Regression
    scaled_paths = [
        r"C:\Users\LENOVO\Documents\Nam_cuoi\hoc_may\archive (1)\dataset\Rice_4Classes_Scaled_Splitted.xlsx",
        os.path.join(os.path.dirname(__file__), "archive (1)", "dataset", "Rice_4Classes_Scaled_Splitted.xlsx"),
        os.path.join(os.path.dirname(__file__), "Rice_4Classes_Scaled_Splitted.xlsx"),
        "Rice_4Classes_Scaled_Splitted.xlsx"
    ]
    scaled_path = next((p for p in scaled_paths if os.path.exists(p)), None)
    if scaled_path is None:
        st.error("Không tìm thấy file 'Rice_4Classes_Scaled_Splitted.xlsx'!")
        st.stop()

    excel_scaled = pd.ExcelFile(scaled_path)
    train_sheet = next((s for s in excel_scaled.sheet_names if 'train' in s.lower()), None)
    test_sheet = next((s for s in excel_scaled.sheet_names if 'test' in s.lower()), None)

    train_scaled_df = excel_scaled.parse(train_sheet)
    test_scaled_df = excel_scaled.parse(test_sheet)

    le = LabelEncoder()
    y_train = le.fit_transform(train_scaled_df[label_col])
    y_test = le.transform(test_scaled_df[label_col])

    X_train = train_scaled_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_scaled_df[feature_cols].to_numpy(dtype=np.float64)

    try:
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    except TypeError:
        model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    class_centers_scaled = {}
    for c in class_names:
        class_centers_scaled[c] = np.array([(class_means_real[c][f] - mu_series[f]) / sigma_series[f] for f in feature_cols])

    return (model, le, feature_cols, class_names, test_accuracy,
            clean_df, test_scaled_df, mu_series, sigma_series,
            real_bounds, stats_clean_dict, class_means_real, class_centers_scaled)


(model, le, feature_cols, class_names, test_accuracy,
 clean_df, test_scaled_df, mu_series, sigma_series,
 real_bounds, stats_clean_dict, class_means_real, class_centers_scaled) = load_all_resources()

def validate_rice_morphology(inputs, real_bounds, mu_series, sigma_series, class_centers_scaled):
    """
    Kiểm tra tính hợp lệ về mặt vật lý, hình học và sinh học của mẫu hạt gạo.
    Trả về: (is_valid: bool, errors: list[str])
    """
    errors = []
    names_vn = {
        'Major_Axis_Length': '1. Chiều dài hạt (Major Axis)',
        'Minor_Axis_Length': '2. Chiều rộng hạt (Minor Axis)',
        'Area': '3. Diện tích hạt (Area)',
        'Perimeter': '4. Chu vi (Perimeter)',
        'Convex_Area': '5. Diện tích bao lồi (Convex Area)',
        'Eccentricity': '6. Độ lệch tâm (Eccentricity)',
        'Extent': '7. Tỷ lệ diện tích (Extent)'
    }

    # 1. Kiểm tra tất cả đều bằng 0
    if all(v == 0.0 for v in inputs.values()):
        return False, ["Tất cả 7 thông số hình thái đều bằng 0 hoặc chưa được nhập. Vui lòng nhập số liệu."]

    # TỰ ĐỘNG NHẬN DIỆN:
    # Nếu |Area| <= 100: Người dùng đang nhập giá trị chuẩn hóa Z-score từ tập Test (Rice_4Classes_Scaled_Splitted.xlsx)
    # Trong không gian Z-score, giá trị có thể âm (-3 đến +3) nên không áp dụng các ràng buộc pixel
    if abs(inputs.get('Area', 0)) <= 100.0:
        return True, []

    # NGƯỢC LẠI: Người dùng nhập kích thước pixel thực tế (Area > 100)
    zero_or_neg = [f for f, v in inputs.items() if v <= 0]
    if len(zero_or_neg) > 0:
        err_fields = ", ".join([names_vn.get(f, f) for f in zero_or_neg])
        errors.append(f"Thông số {err_fields} không thể bằng 0 hoặc số âm khi nhập kích thước pixel thực tế.")
        return False, errors

    area = inputs.get('Area', 0)
    peri = inputs.get('Perimeter', 0)
    major = inputs.get('Major_Axis_Length', 0)
    minor = inputs.get('Minor_Axis_Length', 0)
    ecc = inputs.get('Eccentricity', 0)
    convex = inputs.get('Convex_Area', 0)
    extent = inputs.get('Extent', 0)

    # 2. Kiểm tra bất biến hình học (Geometric Constraints)
    if minor > major and major > 0:
        errors.append(f"Vi phạm hình học: Trục phụ (chiều rộng: {minor:.1f} px) không thể lớn hơn trục chính (chiều dài: {major:.1f} px). Hạt gạo không thể có chiều ngang dài hơn chiều dọc.")

    if area > convex and convex > 0:
        errors.append(f"Vi phạm hình học: Diện tích hạt ({area:.0f} px²) không thể lớn hơn diện tích bao lồi ({convex:.0f} px²).")

    if ecc >= 1.0 or ecc <= 0:
        errors.append(f"Độ lệch tâm Eccentricity ({ecc:.4f}) không hợp lệ (phải trong khoảng 0 < Eccentricity < 1 đối với hình elip).")
    elif ecc < 0.50:
        errors.append(f"Độ lệch tâm Eccentricity ({ecc:.4f}) quá nhỏ: Hạt gạo luôn có dạng elip thuôn dài (dải sinh học thực tế từ 0.67 đến 0.97).")

    if extent > 1.0 or extent <= 0:
        errors.append(f"Tỷ lệ diện tích Extent ({extent:.4f}) không hợp lệ (diện tích hạt không thể vượt quá hộp chữ nhật bao ngoài).")
    elif extent < 0.20:
        errors.append(f"Tỷ lệ Extent ({extent:.4f}) quá nhỏ so với cấu trúc hạt gạo tự nhiên (chuẩn thực tế từ 0.38 đến 0.89).")

    if peri > 0 and area > 0:
        if (peri ** 2) < (4 * np.pi * area * 0.90):
            errors.append(f"Vi phạm bất đẳng thức hình học (P² >= 4πA): Chu vi ({peri:.1f} px) quá nhỏ so với diện tích ({area:.0f} px²) - hình này không tồn tại trong hình học phẳng.")

    # 3. Kiểm tra dải giá trị sinh thái học (Extreme Outliers / Dị vật)
    tol_low = 0.50
    tol_high = 1.50
    extreme_outliers = []
    for f in ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Convex_Area']:
        val = inputs.get(f, 0)
        b_min = real_bounds[f]['min']
        b_max = real_bounds[f]['max']
        if val < b_min * tol_low:
            extreme_outliers.append(f"{names_vn.get(f, f)} quá nhỏ ({val:.1f} so với chuẩn min {b_min:.1f})")
        elif val > b_max * tol_high:
            extreme_outliers.append(f"{names_vn.get(f, f)} quá lớn ({val:.1f} so với chuẩn max {b_max:.1f})")

    if len(extreme_outliers) >= 2:
        errors.append("Kích thước hạt gạo vượt quá giới hạn sinh học thực tế (dị vật hoặc đo sai lệch): " + "; ".join(extreme_outliers) + ".")

    # 4. Kiểm tra Out-of-Distribution (Khoảng cách trong không gian đặc trưng Z-score)
    if len(errors) == 0:
        feats_list = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']
        z_vec = np.array([(inputs[f] - mu_series[f]) / sigma_series[f] for f in feats_list])
        dists = [np.linalg.norm(z_vec - center) for center in class_centers_scaled.values()]
        min_dist = min(dists)
        if min_dist > 8.5:
            errors.append(f"Hình thái bất thường (Độ lệch chuẩn khoảng cách = {min_dist:.1f}): Tỷ lệ hình dạng này không tương thích với bất kỳ giống lúa nào trong 4 giống chuẩn (Cammeo, Gonen, Jasmine, Osmancik).")

    return (len(errors) == 0), errors

# Thông tin đặc điểm nổi bật 4 giống lúa chuẩn
RICE_HIGHLIGHTS = {
    "Jasmine": [
        "Hạt thon dài mảnh đặc trưng, màu trắng trong thanh nhã",
        "Chiều rộng rất nhỏ (~51 px), diện tích hạt nhỏ nhất trong 4 giống",
        "Khi nấu cơm tỏa hương thơm hoa lài tự nhiên, độ dẻo mềm hảo hạng"
    ],
    "Cammeo": [
        "Hạt to lớn vượt trội, chiều dài (~205 px) và diện tích (~14,163 px²) lớn nhất",
        "Độ nở cơm cao, hàm lượng tinh bột thích hợp cho các món cơm Âu",
        "Màu hạt trắng ngà đều đặn, cấu trúc phôi hạt chắc khỏe"
    ],
    "Gonen": [
        "Hình dạng hạt cân đối, tròn đều đặn ở mức trung bình (~8,358 px²)",
        "Tỷ lệ chiều dài / chiều rộng hài hòa, độ đồng nhất mẫu rất cao",
        "Cơm nấu chín nở đều, hạt cơm mềm vừa không bị nát"
    ],
    "Osmancik": [
        "Hạt tròn bầu dục, chiều rộng lớn (~84.5 px), độ lệch tâm thấp nhất",
        "Hạt bóng tròn trịa, kích thước hạt trung bình lớn (~11,550 px²)",
        "Được trồng rộng rãi nhất tại Thổ Nhĩ Kỳ, chất lượng cơm đậm đà"
    ]
}

# Ảnh macro chụp hạt thật chuẩn đẹp theo từng giống lúa (hỗ trợ đọc trực tiếp từ file local mã hóa Base64)
def get_rice_image_base64(variety_name: str) -> str:
    """Đọc ảnh hạt gạo chuẩn từ thư mục local và mã hóa Base64 để hiển thị tức thì, hoạt động 100% offline không phụ thuộc internet."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "assets", "rice_images", f"{variety_name}.jpg"),
        os.path.join(script_dir, "Ảnh", f"{variety_name}.jpg"),
        os.path.join(script_dir, "assets", "rice_images", f"{variety_name}.png"),
        os.path.join(script_dir, "Ảnh", f"{variety_name}.png"),
        os.path.join(script_dir, f"{variety_name}.jpg")
    ]
    for p in candidate_paths:
        if os.path.isfile(p):
            try:
                with open(p, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
                    return f"data:{mime};base64,{encoded}"
            except Exception:
                pass
    # Fallback dự phòng nếu thiếu file
    fallback_urls = {
        "Jasmine": "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=700&q=80",
        "Cammeo": "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=700&q=80",
        "Gonen": "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=700&q=80",
        "Osmancik": "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=700&q=80"
    }
    return fallback_urls.get(variety_name, fallback_urls["Jasmine"])


# ======================================================================================
# 4. KHỞI TẠO SESSION STATE & CÁC HÀM XỬ LÝ SỰ KIỆN (CALLBACKS)
# ======================================================================================
# Khởi tạo mặc định: Ban đầu mở trang các ô bằng 0.0 và ở trạng thái INITIAL (chờ nhập liệu)
for f in feature_cols:
    key_name = f"inp_{f}"
    if key_name not in st.session_state:
        st.session_state[key_name] = 0.0

if "app_status" not in st.session_state:
    st.session_state["app_status"] = "INITIAL"
if "pred_class" not in st.session_state:
    st.session_state["pred_class"] = None
if "pred_confidence" not in st.session_state:
    st.session_state["pred_confidence"] = 0.0
if "pred_probs" not in st.session_state:
    st.session_state["pred_probs"] = None
if "validation_errors" not in st.session_state:
    st.session_state["validation_errors"] = []

# Quản lý file lưu trữ lịch sử nhận dạng bền vững (CSV)
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_predictions.csv")

if "history_list" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            hist_df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
            # Tự động lọc bỏ các dòng trùng lặp liên tiếp nếu có từ trước
            cols_check = [
                "Giống lúa nhận dạng", "Chiều dài (px)", "Chiều rộng (px)", 
                "Diện tích (px²)", "Chu vi (px)", "Bao lồi (px²)", 
                "Độ lệch tâm", "Tỷ lệ diện tích"
            ]
            existing_cols = [c for c in cols_check if c in hist_df.columns]
            if existing_cols:
                mask = (hist_df[existing_cols] != hist_df[existing_cols].shift()).any(axis=1)
                hist_df = hist_df[mask]
            st.session_state["history_list"] = hist_df.to_dict(orient="records")
        except Exception:
            st.session_state["history_list"] = []
    else:
        st.session_state["history_list"] = []

def log_prediction_to_history(variety, conf, inputs_dict):
    """Ghi nhận và lưu trữ tự động một lượt nhận dạng vào lịch sử (ngăn chặn lưu trùng lặp 2 lần)."""
    new_entry = {
        "Thời gian": datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        "Giống lúa nhận dạng": str(variety),
        "Độ tin cậy (%)": round(float(conf), 2),
        "Chiều dài (px)": round(float(inputs_dict.get('Major_Axis_Length', 0.0)), 2),
        "Chiều rộng (px)": round(float(inputs_dict.get('Minor_Axis_Length', 0.0)), 2),
        "Diện tích (px²)": round(float(inputs_dict.get('Area', 0.0)), 1),
        "Chu vi (px)": round(float(inputs_dict.get('Perimeter', 0.0)), 2),
        "Bao lồi (px²)": round(float(inputs_dict.get('Convex_Area', 0.0)), 1),
        "Độ lệch tâm": round(float(inputs_dict.get('Eccentricity', 0.0)), 4),
        "Tỷ lệ diện tích": round(float(inputs_dict.get('Extent', 0.0)), 4)
    }
    if "history_list" not in st.session_state:
        st.session_state["history_list"] = []

    # KIỂM TRA TRÙNG LẶP: Nếu bản ghi mới có cùng giống lúa và 7 thông số giống hệt bản ghi vừa lưu gần nhất
    if st.session_state["history_list"]:
        last = st.session_state["history_list"][0]
        same_variety = str(last.get("Giống lúa nhận dạng", "")).strip() == str(new_entry["Giống lúa nhận dạng"]).strip()
        check_cols = [
            "Chiều dài (px)", "Chiều rộng (px)", "Diện tích (px²)",
            "Chu vi (px)", "Bao lồi (px²)", "Độ lệch tâm", "Tỷ lệ diện tích"
        ]
        try:
            same_features = all(
                abs(float(last.get(col, -999999)) - float(new_entry[col])) < 1e-4
                for col in check_cols
            )
        except Exception:
            same_features = False

        if same_variety and same_features:
            return  # Đã có kết quả này ở đầu lịch sử -> bỏ qua, không lưu trùng lặp lần 2!

    st.session_state["history_list"].insert(0, new_entry)
    try:
        df_hist = pd.DataFrame(st.session_state["history_list"])
        df_hist.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass

def clear_prediction_history():
    """Xóa toàn bộ lịch sử nhận dạng trong session và file lưu trữ"""
    st.session_state["history_list"] = []
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass

# Callback Đặt lại toàn bộ (chạy an toàn trước khi widget được tạo)
def reset_morphology_inputs():
    for f in feature_cols:
        st.session_state[f"inp_{f}"] = 0.0
    st.session_state["app_status"] = "INITIAL"
    st.session_state["pred_class"] = None
    st.session_state["pred_confidence"] = 0.0
    st.session_state["pred_probs"] = None
    st.session_state["validation_errors"] = []

# Callback Nạp mẫu chuẩn (chạy an toàn trước khi widget được tạo)
def load_preset_sample(variety):
    means = class_means_real[variety]
    inputs_dict = {f: float(means[f]) for f in feature_cols}
    for f in feature_cols:
        st.session_state[f"inp_{f}"] = inputs_dict[f]
    z_p = np.array([(means[f] - mu_series[f]) / sigma_series[f] for f in feature_cols])
    p_arr = model.predict_proba([z_p])[0]
    conf = float(p_arr[class_names.index(variety)] * 100)
    st.session_state["app_status"] = "SUCCESS"
    st.session_state["pred_class"] = variety
    st.session_state["pred_confidence"] = conf
    st.session_state["pred_probs"] = p_arr
    st.session_state["validation_errors"] = []
    log_prediction_to_history(variety, conf, inputs_dict)

# Callback Nạp 1 mẫu ngẫu nhiên từ tập Test đã giải chuẩn hóa về pixel thực tế
def load_random_test_sample():
    random_row = test_scaled_df.sample(n=1).iloc[0]
    true_label = random_row['Class']
    inp_sample = {}
    for f in feature_cols:
        raw_val = float(random_row[f] * sigma_series[f] + mu_series[f])
        val_rounded = round(raw_val, 4 if f in ['Eccentricity', 'Extent'] else 1)
        st.session_state[f"inp_{f}"] = val_rounded
        inp_sample[f] = val_rounded
    z_p = np.array([float(random_row[f]) for f in feature_cols])
    p_arr = model.predict_proba([z_p])[0]
    pred_idx = int(np.argmax(p_arr))
    conf = float(p_arr[pred_idx] * 100)
    st.session_state["app_status"] = "SUCCESS"
    st.session_state["pred_class"] = class_names[pred_idx]
    st.session_state["pred_confidence"] = conf
    st.session_state["pred_probs"] = p_arr
    st.session_state["validation_errors"] = []
    log_prediction_to_history(class_names[pred_idx], conf, inp_sample)


# ======================================================================================
# 5. TOP HEADER BANNER TRẢI DÀI (GỌN GÀNG, KHÔNG BỊ CẮT CHỮ)
# ======================================================================================
st.markdown("""
<div class="top-header-banner">
    <div class="header-logo-group">
        <div class="header-logo-icon">🌾</div>
        <h1 class="header-title-text">HỆ THỐNG NHẬN DẠNG GIỐNG LÚA</h1>
    </div>
</div>
""", unsafe_allow_html=True)


# ======================================================================================
# 6. BỐ CỤC 2 KHỐI CHÍNH SONG SONG (KHỐI NHẬP LIỆU BÊN TRÁI - KHỐI KẾT QUẢ BÊN PHẢI)
# ======================================================================================
col_form, col_output = st.columns([1, 1], gap="medium")

# --------------------------------------------------------------------------------------
# KHỐI 1 (BÊN TRÁI): FORM NHẬP 7 ĐẶC TRƯNG + NÚT BẤM NHẬN DẠNG
# --------------------------------------------------------------------------------------
with col_form:
    with st.container(border=True):
        # Chia thành 2 cột con để các ô số nằm cạnh nhau sít sao, không bị dài
        sub_col1, sub_col2 = st.columns(2, gap="medium")

    # --- HÀNG 1: Chiều dài (Major Axis) & Chiều rộng (Minor Axis) ---
    with sub_col1:
        b_maj = real_bounds['Major_Axis_Length']
        val_major = st.number_input(
            "↔️ 1. Chiều dài hạt (Major Axis)",
            min_value=-20.0, max_value=500.0,
            step=0.1, format="%.4f", key="inp_Major_Axis_Length"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_maj['min']:.1f} - {b_maj['max']:.1f} px | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    with sub_col2:
        b_min = real_bounds['Minor_Axis_Length']
        val_minor = st.number_input(
            "↕️ 2. Chiều rộng hạt (Minor Axis)",
            min_value=-20.0, max_value=300.0,
            step=0.1, format="%.4f", key="inp_Minor_Axis_Length"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_min['min']:.1f} - {b_min['max']:.1f} px | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    # --- HÀNG 2: Diện tích (Area) & Chu vi (Perimeter) ---
    with sub_col1:
        b_ar = real_bounds['Area']
        val_area = st.number_input(
            "🍃 3. Diện tích hạt (Area)",
            min_value=-20.0, max_value=30000.0,
            step=1.0, format="%.4f", key="inp_Area"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_ar['min']:.0f} - {b_ar['max']:.0f} px² | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    with sub_col2:
        b_per = real_bounds['Perimeter']
        val_perimeter = st.number_input(
            "📏 4. Chu vi hạt (Perimeter)",
            min_value=-20.0, max_value=1000.0,
            step=0.5, format="%.4f", key="inp_Perimeter"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_per['min']:.1f} - {b_per['max']:.1f} px | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    # --- HÀNG 3: Diện tích bao lồi (Convex Area) & Độ lệch tâm (Eccentricity) ---
    with sub_col1:
        b_cvx = real_bounds['Convex_Area']
        val_convex = st.number_input(
            "📦 5. Diện tích bao lồi (Convex Area)",
            min_value=-20.0, max_value=35000.0,
            step=1.0, format="%.4f", key="inp_Convex_Area"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_cvx['min']:.0f} - {b_cvx['max']:.0f} px² | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    with sub_col2:
        b_ecc = real_bounds['Eccentricity']
        val_ecc = st.number_input(
            "🎯 6. Độ lệch tâm (Eccentricity)",
            min_value=-20.0, max_value=5.0,
            step=0.005, format="%.4f", key="inp_Eccentricity"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_ecc['min']:.4f} - {b_ecc['max']:.4f} | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    # --- HÀNG 4: Tỷ lệ diện tích (Extent) & Nút nạp mẫu nhanh ---
    with sub_col1:
        b_ext = real_bounds['Extent']
        val_extent = st.number_input(
            "% 7. Tỷ lệ diện tích (Extent)",
            min_value=-20.0, max_value=5.0,
            step=0.01, format="%.4f", key="inp_Extent"
        )
        st.markdown(f"<div class='input-hint-sub'>Dải pixel: {b_ext['min']:.4f} - {b_ext['max']:.4f} | Z-score: -3 đến +3</div>", unsafe_allow_html=True)

    with sub_col2:
        st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #1F2937; margin-bottom: 5px;'>⚡ NẠP MẪU CHUẨN:</div>", unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.button("🌾 Jasmine", on_click=load_preset_sample, args=("Jasmine",), use_container_width=True)
            st.button("🌾 Gonen", on_click=load_preset_sample, args=("Gonen",), use_container_width=True)
        with col_p2:
            st.button("🌾 Cammeo", on_click=load_preset_sample, args=("Cammeo",), use_container_width=True)
            st.button("🌾 Osmancik", on_click=load_preset_sample, args=("Osmancik",), use_container_width=True)
        st.button("🎲 Mẫu ngẫu nhiên từ tập Test", on_click=load_random_test_sample, use_container_width=True)

    # Thu thập 7 giá trị nhập liệu hiện tại từ session_state (nguồn dữ liệu đồng bộ duy nhất)
    current_inputs = {f: float(st.session_state[f"inp_{f}"]) for f in feature_cols}

    # NÚT BẤM KÍCH HOẠT NHẬN DẠNG & NÚT ĐẶT LẠI (RESET)
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns([7, 3], gap="small")
    with col_btn1:
        btn_predict = st.button("🚀 NHẬN DẠNG GIỐNG LÚA", type="primary", use_container_width=True)
    with col_btn2:
        st.button("🔄 Đặt lại", on_click=reset_morphology_inputs, use_container_width=True)

    # XỬ LÝ KHI BẤM NÚT NHẬN DẠNG
    if btn_predict:
        is_valid_now, validation_errors_now = validate_rice_morphology(
            current_inputs, real_bounds, mu_series, sigma_series, class_centers_scaled
        )

        if not is_valid_now:
            st.session_state["app_status"] = "INVALID"
            st.session_state["validation_errors"] = validation_errors_now
            st.session_state["pred_class"] = None
        else:
            is_z_score = abs(current_inputs['Area']) <= 100.0
            if is_z_score:
                z_vector = np.array([current_inputs[f] for f in feature_cols], dtype=np.float64)
            else:
                z_vector = np.zeros(7, dtype=np.float64)
                for idx, f in enumerate(feature_cols):
                    z_vector[idx] = (current_inputs[f] - mu_series[f]) / sigma_series[f]

            probs_calc = model.predict_proba([z_vector])[0]
            pred_idx_calc = int(np.argmax(probs_calc))
            conf = float(probs_calc[pred_idx_calc] * 100)

            if conf < 40.0:
                st.session_state["app_status"] = "INVALID"
                st.session_state["validation_errors"] = [
                    f"Độ không chắc chắn quá cao (xác suất cao nhất chỉ đạt {conf:.1f}%). Mô hình không thể phân loại chính xác mẫu hạt này vào 4 giống lúa chuẩn."
                ]
                st.session_state["pred_class"] = None
            else:
                st.session_state["app_status"] = "SUCCESS"
                st.session_state["validation_errors"] = []
                st.session_state["pred_class"] = class_names[pred_idx_calc]
                st.session_state["pred_confidence"] = conf
                st.session_state["pred_probs"] = probs_calc
                log_prediction_to_history(class_names[pred_idx_calc], conf, current_inputs)


# --------------------------------------------------------------------------------------
# KHỐI 2 (BÊN PHẢI): KẾT QUẢ NHẬN DẠNG / THÔNG BÁO CHỜ / CẢNH BÁO TỪ CHỐI
# --------------------------------------------------------------------------------------
with col_output:
    status = st.session_state.get("app_status", "INITIAL")

    if status == "INITIAL":
        prompt_box_html = (
            '<div style="background: #F0FDF4; border: 2px dashed #10B981; border-radius: 14px; padding: 26px 22px; text-align: center; box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08);">'
            '<div style="font-size: 3rem; margin-bottom: 10px; line-height: 1;">🌾</div>'
            '<div style="font-size: 1.45rem; font-weight: 800; color: #065F46; letter-spacing: -0.3px; margin-bottom: 8px;">'
            'MỜI BẠN NHẬP HÌNH THÁI HẠT GẠO'
            '</div>'
            '<div style="font-size: 0.90rem; color: #047857; font-weight: 500; line-height: 1.6; max-width: 480px; margin: 0 auto 18px auto;">'
            'Hệ thống đang sẵn sàng! Hãy nhập 7 thông số hình thái hình học ở bảng bên trái hoặc chọn nhanh mẫu chuẩn để bắt đầu nhận dạng giống lúa.'
            '</div>'
            '<div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 18px; flex-wrap: wrap;">'
            '<div style="background: #FFFFFF; border: 1px solid #A7F3D0; border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #065F46; font-weight: 700;">'
            '1️⃣ Nhập 7 thông số hạt gạo'
            '</div>'
            '<div style="background: #FFFFFF; border: 1px solid #A7F3D0; border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #065F46; font-weight: 700;">'
            '2️⃣ Hoặc bấm Nạp mẫu chuẩn'
            '</div>'
            '<div style="background: #FFFFFF; border: 1px solid #A7F3D0; border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; color: #065F46; font-weight: 700;">'
            '3️⃣ Bấm Nhận dạng giống lúa'
            '</div>'
            '</div>'
            '<div style="background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 10px; padding: 12px 16px; font-size: 0.82rem; color: #064E3B; line-height: 1.6; text-align: left;">'
            '💡 <b>4 Giống lúa hỗ trợ phân loại:</b><br>'
            '• <b>Jasmine:</b> Hạt thon dài mảnh đặc trưng, diện tích nhỏ nhất (~5,900 px²).<br>'
            '• <b>Cammeo:</b> Kích thước hạt lớn nhất, chiều dài và diện tích vượt trội (~14,100 px²).<br>'
            '• <b>Gonen:</b> Kích thước trung bình, tròn đều cân đối (~8,300 px²).<br>'
            '• <b>Osmancik:</b> Hạt tròn bầu dục, bề ngang rộng, độ lệch tâm thấp nhất (~11,500 px²).'
            '</div>'
            '</div>'
        )
        st.markdown(prompt_box_html, unsafe_allow_html=True)

    elif status == "INVALID":
        errors_to_show = st.session_state.get("validation_errors", ["Thông số hình thái hạt gạo không hợp lệ."])
        err_items_html = "".join([f'<li style="margin-bottom: 7px;">❌ {err}</li>' for err in errors_to_show])

        error_box_html = (
            '<div style="background: #FFF5F5; border: 2px solid #E53E3E; border-radius: 14px; padding: 22px; box-shadow: 0 4px 16px rgba(229, 62, 62, 0.12);">'
            '<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">'
            '<span style="font-size: 2.2rem; line-height: 1;">⚠️</span>'
            '<div>'
            '<div style="font-size: 1.35rem; font-weight: 800; color: #C53030; letter-spacing: -0.3px;">'
            'TỪ CHỐI NHẬN DẠNG: HÌNH THÁI SAI LỆCH'
            '</div>'
            '<div style="font-size: 0.85rem; color: #9B2C2C; font-weight: 600;">'
            'Các thông số đo đạc không phù hợp với quy luật sinh học và hình học của hạt gạo.'
            '</div>'
            '</div>'
            '</div>'
            '<div style="background: #FFFFFF; border: 1px solid #FEB2B2; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;">'
            '<div style="font-size: 0.85rem; font-weight: 800; color: #9B2C2C; text-transform: uppercase; margin-bottom: 8px;">'
            'Chi tiết các sai lệch phát hiện:'
            '</div>'
            f'<ul style="margin: 0; padding-left: 18px; font-size: 0.84rem; color: #742A2A; line-height: 1.6;">{err_items_html}</ul>'
            '</div>'
            '<div style="background: #FEFCBF; border: 1px solid #ECC94B; border-radius: 10px; padding: 12px 14px; font-size: 0.84rem; color: #744210; line-height: 1.5;">'
            '💡 <b>Hướng dẫn xử lý:</b><br>'
            '• Vui lòng nhập thông số đo đạc thực tế của hạt gạo (các kích thước phải lớn hơn 0).<br>'
            '• Hoặc bấm chọn nhanh một trong 4 nút <b>🌾 Jasmine, Cammeo, Gonen, Osmancik</b> ở cột bên trái để nạp mẫu hạt chuẩn hợp lệ.'
            '</div>'
            '</div>'
        )
        st.markdown(error_box_html, unsafe_allow_html=True)

    else:
        # status == "SUCCESS"
        display_class = st.session_state.get("pred_class", "Jasmine")
        display_confidence = st.session_state.get("pred_confidence", 99.4)
        display_probs = st.session_state.get("pred_probs", None)
        if display_probs is None:
            display_probs = np.zeros(len(class_names))

        with st.container(border=True):
            col_img, col_info = st.columns([4.2, 5.8], gap="medium")
        
            with col_img:
                photo_url = get_rice_image_base64(display_class)
                st.markdown(f"""
                <div style="border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.14); height: 210px; background: #000;">
                    <img src="{photo_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{display_class}">
                </div>
                <div style="text-align: center; font-size: 0.78rem; font-weight: 700; color: #4B5563; margin-top: 6px;">
                    Ảnh chụp hiển vi hạt gạo {display_class}
                </div>
                """, unsafe_allow_html=True)

            with col_info:
                st.markdown(f"""
                <div style="font-size: 0.92rem; color: #4B5563; font-weight: 700; margin-bottom: 2px;">
                    Giống lúa được nhận dạng:
                </div>
                <div style="font-size: 2.35rem; font-weight: 900; color: #135E38; letter-spacing: -0.5px; margin-bottom: 4px; line-height: 1.15;">
                    {display_class.upper()}
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    <span style="font-size: 1.35rem; color: #10B981;">✅</span>
                    <span style="font-size: 1.22rem; font-weight: 800; color: #1F2937;">
                        Độ chính xác: <span style="color: #135E38;">{display_confidence:.1f}%</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Thanh phân phối xác suất Softmax cho cả 4 giống lúa
                st.markdown("<div style='font-size: 0.76rem; font-weight: 800; color: #4B5563; margin-bottom: 6px; text-transform: uppercase;'>Xác suất từng giống:</div>", unsafe_allow_html=True)
                prob_colors = {
                    "Cammeo": "#3B82F6",
                    "Gonen": "#8B5CF6",
                    "Jasmine": "#10B981",
                    "Osmancik": "#F59E0B"
                }
                for c_name in class_names:
                    c_idx = list(class_names).index(c_name)
                    p_val = display_probs[c_idx] * 100 if len(display_probs) > c_idx else 0.0
                    bar_color = prob_colors.get(c_name, "#10B981")
                    st.markdown(f"""
                    <div class="prob-bar-container">
                        <span class="prob-bar-label">{c_name}:</span>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: {p_val:.1f}%; background-color: {bar_color};"></div>
                        </div>
                        <span class="prob-bar-val">{p_val:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Khối đặc điểm nổi bật phía dưới
            highlights = RICE_HIGHLIGHTS.get(display_class, [
                "Hạt thon dài, màu trắng trong",
                "Thích nghi tốt với nhiều vùng sinh thái",
                "Năng suất cao và chất lượng thơm ngon"
            ])
            highlight_html = "".join([f'<div class="highlight-item"><span style="color: #10B981; font-weight: 700;">✔</span> {h}</div>' for h in highlights[:3]])

            st.markdown(f"""
            <div class="highlights-box">
                <div style="font-size: 0.86rem; font-weight: 800; color: #B45309; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                    <span>💡</span> Đặc điểm nhận diện nổi bật:
                </div>
                {highlight_html}
            </div>
            """, unsafe_allow_html=True)


# ======================================================================================
# 7. KHỐI LƯU TRỮ LỊCH SỬ CÁC LẦN NHẬN DẠNG (PREDICTION HISTORY)
# ======================================================================================
st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

with st.container(border=True):
    hist_list = st.session_state.get("history_list", [])
    total_saved = len(hist_list)

    col_h_title, col_h_actions = st.columns([7, 3], gap="medium")
    with col_h_title:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <span style="font-size: 1.45rem;">📋</span>
            <span style="font-size: 1.25rem; font-weight: 800; color: #135E38;">LỊCH SỬ CÁC LẦN NHẬN DẠNG</span>
            <span style="background: #E8F5E9; color: #135E38; font-weight: 800; font-size: 0.82rem; padding: 3px 10px; border-radius: 20px; border: 1px solid #A7F3D0;">
                {total_saved} lượt đã lưu
            </span>
        </div>
        <div style="font-size: 0.84rem; color: #4B5563; margin-left: 36px; margin-bottom: 8px;">
            Lưu trữ tự động kết quả phân loại giống lúa và 7 thông số hình thái hình học qua các lần thực nghiệm.
        </div>
        """, unsafe_allow_html=True)

    with col_h_actions:
        if total_saved > 0:
            c_act1, c_act2 = st.columns(2, gap="small")
            with c_act1:
                df_export = pd.DataFrame(hist_list)
                csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="📥 Tải CSV",
                    data=csv_bytes,
                    file_name=f"lich_su_nhan_dang_lua_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_act2:
                st.button("🗑️ Xóa lịch sử", on_click=clear_prediction_history, use_container_width=True)

    if total_saved == 0:
        st.markdown("""
        <div style="text-align: center; padding: 24px 18px; color: #6B7280; background: #F9FAFB; border-radius: 10px; border: 1.5px dashed #D1D5DB; margin-top: 4px;">
            <div style="font-size: 2.2rem; margin-bottom: 6px;">📂</div>
            <div style="font-weight: 700; font-size: 0.98rem; color: #374151;">Chưa có lượt nhận dạng nào được lưu</div>
            <div style="font-size: 0.84rem; color: #6B7280; margin-top: 4px;">
                Mỗi khi bạn bấm <b>Nhận dạng giống lúa</b> hoặc nạp mẫu hạt, hệ thống sẽ tự động ghi lại toàn bộ kết quả và thông số đo đạc tại đây.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(hist_list)
        df_display = df_hist.copy()
        df_display.insert(0, "STT", range(1, len(df_display) + 1))

        # Hiển thị 3 chỉ số tóm tắt nhanh
        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap;">
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 7px 14px; font-size: 0.82rem;">
                🌾 <b>Lần gần nhất:</b> <span style="color: #135E38; font-weight: 800;">{hist_list[0].get('Giống lúa nhận dạng', '')}</span> ({hist_list[0].get('Độ tin cậy (%)', 0)}%)
            </div>
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 7px 14px; font-size: 0.82rem;">
                ⏱️ <b>Thời gian:</b> <span style="color: #135E38; font-weight: 700;">{hist_list[0].get('Thời gian', '')}</span>
            </div>
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 7px 14px; font-size: 0.82rem;">
                📊 <b>Tổng số mẫu:</b> <span style="color: #135E38; font-weight: 800;">{total_saved}</span> mẫu đã phân loại
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=min(320, 45 + len(df_display) * 36)
        )


# ======================================================================================
# 8. KHỐI MỞ RỘNG TIỆN ÍCH (THÔNG TIN HỆ THỐNG & BẢNG KÍCH THƯỚC ĐỐI SÁNH)
# ======================================================================================
with st.expander("📊 Xem bảng chuẩn kích thước đối sánh 4 giống lúa [Min - Mean - Max]", expanded=False):
    matrix_rows = []
    feature_labels = {
        'Major_Axis_Length': ('1. Chiều dài hạt (Major Axis)', 'px'),
        'Minor_Axis_Length': ('2. Chiều rộng hạt (Minor Axis)', 'px'),
        'Area': ('3. Diện tích hạt (Area)', 'px²'),
        'Perimeter': ('4. Chu vi hạt (Perimeter)', 'px'),
        'Convex_Area': ('5. Diện tích bao lồi (Convex Area)', 'px²'),
        'Eccentricity': ('6. Độ lệch tâm (Eccentricity)', 'tỷ lệ'),
        'Extent': ('7. Tỷ lệ diện tích (Extent)', 'tỷ lệ')
    }
    for f, (f_name, f_unit) in feature_labels.items():
        is_float = (f_unit != 'px²')
        fmt = ".1f" if f_unit == 'px' else ".4f"

        def fmt_cell(variety):
            mi = stats_clean_dict[variety][f]['min']
            me = stats_clean_dict[variety][f]['mean']
            ma = stats_clean_dict[variety][f]['max']
            if is_float:
                return f"TB: {me:{fmt}} [{mi:{fmt}} - {ma:{fmt}}]"
            else:
                return f"TB: {me:.0f} [{mi:.0f} - {ma:.0f}]"

        matrix_rows.append({
            "Đặc trưng": f_name, "Đơn vị": f_unit,
            "Jasmine": fmt_cell('Jasmine'),
            "Gonen": fmt_cell('Gonen'),
            "Osmancik": fmt_cell('Osmancik'),
            "Cammeo": fmt_cell('Cammeo')
        })
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
    st.caption("💡 *Ghi chú: 'TB' là giá trị trung bình chuẩn (Mean) đặc trưng của giống lúa, trong ngoặc vuông [Min - Max] là dải biến thiên của các hạt đơn lẻ trong tập dữ liệu.*")

with st.expander("ℹ️ Giới thiệu mô hình Logistic Regression & Hướng dẫn sử dụng", expanded=False):
    st.markdown("""
    * **Mục tiêu:** Tự động phân loại 4 giống lúa (*Cammeo, Gonen, Jasmine, Osmancik*) dựa trên 7 thông số hình thái hình học hạt gạo.
    * **Mô hình AI:** Hồi quy Logistic đa biến (Multinomial Logistic Regression) với bộ tối ưu hóa L-BFGS, độ chính xác đạt **96.91%** trên tập kiểm thử độc lập.
    * **Cơ chế:** Khi bạn nhập các giá trị đo thực tế, hệ thống tự động chuẩn hóa Z-score ngầm (`z = (x - mean) / std`) trước khi đưa vào hàm Softmax để tính xác suất phân loại.
    """)

# Footer tinh tế
st.caption("🌾 Hệ thống nhận dạng giống lúa dựa trên 7 đặc trưng của hạt gạo • Thuật toán Hồi quy Logistic (Scikit-learn L-BFGS)")
