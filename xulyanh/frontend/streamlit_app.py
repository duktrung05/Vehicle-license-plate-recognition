"""
🚗 License Plate Recognition System
Main Streamlit Application Entry Point
 
Run:
    streamlit run frontend/streamlit_app.py
"""
 
import sys
from pathlib import Path
 
# Force reload local modules on each run to prevent Streamlit caching outdated modules in RAM
for mod in list(sys.modules.keys()):
    if mod.startswith("src.") or mod.startswith("pipeline.") or mod.startswith("recognition.") or mod.startswith("preprocessing.") or mod.startswith("postprocessing.") or mod.startswith("detection.") or mod.startswith("storage."):
        del sys.modules[mod]
 
 
import streamlit as st
from PIL import Image
 
 
# =============================================================================
# PATH SETUP
# =============================================================================
 
ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
 
sys.path.append(str(ROOT))
sys.path.append(str(SRC_DIR))
 
 
# =============================================================================
# PAGE CONFIG
# =============================================================================
 
st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
 
# =============================================================================
# CSS
# =============================================================================
 
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
 
    .stApp {
        background-color: #0A0E1A;
        color: #E0E8FF;
    }
 
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 1px;
    }
    
    h2, h3 {
        margin-top: 2.5rem !important;
        margin-bottom: 1.2rem !important;
    }
 
    .metric-card {
        background: linear-gradient(135deg, #0F1629 0%, #162040 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
        min-height: 150px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: #00D4FF;
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.2);
        background: linear-gradient(135deg, #121B33 0%, #1B2952 100%);
    }
 
    .metric-value {
        font-size: 2.3rem;
        font-family: 'JetBrains Mono', monospace;
        color: #00D4FF;
        font-weight: 600;
    }
 
    .metric-label {
        color: #A3B3CC;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
 
    .plate-display {
        background: #1A1A2E;
        border: 2px solid #FFD700;
        border-radius: 8px;
        padding: 15px 25px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFD700;
        text-align: center;
        letter-spacing: 3px;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
        display: inline-block;
    }

    /* Premium Custom Button Styling */
    div.stButton > button {
        background-color: #00D4FF !important;
        color: #0A0E1A !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        background-color: #00B3DB !important;
        color: #0A0E1A !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.35) !important;
    }

    /* System Status Card Styling */
    .status-card {
        background: #0F1629;
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 80px;
        transition: all 0.3s ease;
    }
    
    .status-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .status-title {
        color: #8892A0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    
    .status-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #E0E8FF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Animated Pulse Dot */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .green-dot {
        background-color: #00FF66;
        box-shadow: 0 0 8px #00FF66;
        animation: pulse-green 2s infinite;
    }
    
    .yellow-dot {
        background-color: #FFAA00;
        box-shadow: 0 0 8px #FFAA00;
        animation: pulse-yellow 2s infinite;
    }
    
    .red-dot {
        background-color: #FF3366;
        box-shadow: 0 0 8px #FF3366;
        animation: pulse-red 2s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 102, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(0, 255, 102, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 102, 0); }
    }
    
    @keyframes pulse-yellow {
        0% { box-shadow: 0 0 0 0 rgba(255, 170, 0, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(255, 170, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 170, 0, 0); }
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(255, 51, 102, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.5rem !important;
        }
        .metric-card {
            padding: 12px !important;
            min-height: 120px !important;
        }
        .status-value {
            font-size: 0.95rem !important;
        }
    }

    /* Proportional centered image styling */
    .stImage, div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        margin: 1rem auto !important;
    }
    .stImage img, div[data-testid="stImage"] img {
        max-width: 100% !important;
        max-height: 75vh !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 12px;
        border: 2px solid rgba(0, 212, 255, 0.25);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .stImage img:hover, div[data-testid="stImage"] img:hover {
        border-color: #00D4FF;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.25);
    }

    /* HUD Panel on the right of the image */
    .hud-panel-right {
        background: rgba(15, 22, 41, 0.85);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 255, 0.15);
        transition: all 0.3s ease;
    }
    .hud-panel-right:hover {
        border-color: #00D4FF;
        box-shadow: 0 4px 25px rgba(0, 212, 255, 0.25);
        transform: translateY(-2px);
    }
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .hud-badge {
        background: #00D4FF;
        color: #0A0E1A;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 4px 12px;
        border-radius: 6px;
    }
    .hud-status {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 3px 10px;
        border: 1.5px solid;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .hud-plate-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #FFFFFF;
        background: #090D1A;
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 16px 10px;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.8);
    }

    /* Scrollable container for HUD cards matching image height */
    .hud-container-scrollable {
        max-height: 68vh;
        overflow-y: auto;
        padding-right: 8px;
        margin-top: 5px;
    }
    /* Style scrollbar for HUD container */
    .hud-container-scrollable::-webkit-scrollbar {
        width: 6px;
    }
    .hud-container-scrollable::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
        border-radius: 4px;
    }
    .hud-container-scrollable::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.3);
        border-radius: 4px;
    }
    .hud-container-scrollable::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.6);
    }

    /* Constrain video height to fit on screen and avoid scrolling */
    div[data-testid="stVideo"] video, video {
        max-height: 48vh !important;
        width: auto !important;
        margin: 0 auto;
        display: block;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)
 
 
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
 
def safe_rerun():
    """Hỗ trợ rerun cho nhiều version Streamlit."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
 
 
def go_to_page(page_name: str):
    st.session_state["current_page"] = page_name
    safe_rerun()
 
 
def check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False
 
 
def render_header():
    st.markdown("# Hệ thống Nhận diện Biển số xe")
    st.divider()
 
 
def render_system_status():
    st.subheader("🧩 Trạng thái Hệ thống")
 
    col1, col2, col3, col4 = st.columns(4)
 
    # Determine statuses
    cv2_status = "Sẵn sàng" if check_import("cv2") else "Thiếu"
    cv2_color = "green" if cv2_status == "Sẵn sàng" else "red"
    
    ocr_status = "Sẵn sàng" if check_import("easyocr") else "Phụ trợ (Fallback)"
    ocr_color = "green" if ocr_status == "Sẵn sàng" else "yellow"
    
    yolo_status = "Sẵn sàng" if check_import("ultralytics") else "Đường bao (Contour)"
    yolo_color = "green" if yolo_status == "Sẵn sàng" else "yellow"
    
    model_path = ROOT / "models" / "yolov8" / "best.pt"
    if model_path.is_file():
        model_status = "Sẵn sàng"
        model_color = "green"
    elif model_path.exists() and model_path.is_dir():
        model_status = "Thư mục"
        model_color = "yellow"
    else:
        model_status = "Thiếu"
        model_color = "red"

    def status_card_html(title, status, color_class):
        return f"""
        <div class="status-card">
            <div class="status-title">{title}</div>
            <div class="status-value">
                <span class="status-dot {color_class}-dot"></span>
                {status}
            </div>
        </div>
        """

    with col1:
        st.markdown(status_card_html("OpenCV", cv2_status, cv2_color), unsafe_allow_html=True)
 
    with col2:
        st.markdown(status_card_html("EasyOCR", ocr_status, ocr_color), unsafe_allow_html=True)
 
    with col3:
        st.markdown(status_card_html("YOLOv8 Library", yolo_status, yolo_color), unsafe_allow_html=True)
 
    with col4:
        st.markdown(status_card_html("Tệp mô hình", model_status, model_color), unsafe_allow_html=True)
        if model_path.exists() and model_path.is_dir():
            st.warning(
                "`models/yolov8/best.pt` đang là thư mục, chưa phải file model YOLO `.pt` thật."
            )
 
    st.divider()
 
 
# =============================================================================
# PAGES
# =============================================================================
 
def render_home():
    render_header()
    render_system_status()
 
    st.subheader("📋 Tính năng hệ thống")
 
    pages = [
        ("📷", "Nhận diện qua Ảnh", "Tải ảnh lên để tự động nhận diện & đọc biển số"),
        ("🎬", "Nhận diện qua Video", "Xử lý file video và trích xuất tất cả biển số xe"),
        ("📊", "Lịch sử & Nhật ký", "Xem danh sách lịch sử nhận diện và tìm kiếm"),
    ]
 
    cols = st.columns(len(pages))
 
    for col, (icon, title, desc) in zip(cols, pages):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="
                        font-family:'Rajdhani', sans-serif;
                        font-size:1.1rem;
                        font-weight:600;
                        color:#E0E8FF;
                        margin:8px 0 4px;
                    ">
                        {title}
                    </div>
                    <div style="color:#6B7A8D;font-size:0.8rem">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            if st.button(f"➜ Mở tính năng", key=f"open_{title}", use_container_width=True):
                go_to_page(title)
 
    st.divider()
 
    st.subheader("📊 Thống kê nhanh")
    try:
        import datetime
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        st.markdown(f"<div style='color:#8892A0; font-size:0.85rem; margin-top:-0.8rem; margin-bottom:1rem;'>🕒 Cập nhật lúc: {now_str} (Dữ liệu thời gian thực)</div>", unsafe_allow_html=True)
    except Exception:
        pass
 
    try:
        from storage.database import PlateDatabase
 
        db = PlateDatabase()
        stats = db.get_stats()
 
        c1, c2, c3, c4 = st.columns(4)
 
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get("total_detections", 0)}</div>
                    <div class="metric-label">Tổng lượt nhận diện</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get("today_detections", 0)}</div>
                    <div class="metric-label">Trong ngày</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
        with c3:
            validation_rate = stats.get("validation_rate", 0)
            valid_count = stats.get("valid_plates", 0)
            total_count = stats.get("total_detections", 0)
            rate_str = f"{validation_rate:.1%} ({valid_count}/{total_count})" if total_count > 0 else "0% (0/0)"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.8rem;">{rate_str}</div>
                    <div class="metric-label">Tỉ lệ biển hợp lệ</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
        with c4:
            avg_confidence = stats.get("avg_confidence", 0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_confidence:.2f}</div>
                    <div class="metric-label">Độ tin cậy TB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    except Exception:
        st.info("📦 Database chưa khởi tạo hoặc chưa có dữ liệu.")
 
    st.divider()
 
#     st.subheader("🔄 Pipeline Overview")
 
#     st.markdown(
#         """
# ```text
# Input (Image / Video / Camera)
#         │
#         ▼
# ┌─────────────────┐
# │  YOLOv8 Detect  │  → Detect license plate bounding boxes
# └────────┬────────┘
#          │
#          ▼
# ┌─────────────────┐
# │  Preprocessor   │  → Resize, denoise, deskew, CLAHE
# └────────┬────────┘
#          │
#          ▼
# ┌─────────────────┐
# │   OCR Engine    │  → EasyOCR / Tesseract
# └────────┬────────┘
#          │
#          ▼
# ┌─────────────────┐
# │  Plate Format   │  → Validate & format biển số VN
# └────────┬────────┘
#          │
#          ▼
# ┌─────────────────┐
# │  SQLite Storage │  → Lưu log, timestamp, tỉnh/thành
# └─────────────────┘
 
# ```
# """,
#     unsafe_allow_html=True,
# )
 
 
def render_image_recognition():
    render_header()
 
    st.subheader("📷 Nhận diện qua Ảnh")
    st.write("Kéo thả hoặc chọn ảnh biển số để kiểm tra nhận diện.")
 
    # ── Sidebar settings ──────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**⚙️ Cài đặt Ảnh**")
        conf_threshold = st.slider(
            "Ngưỡng độ tin cậy", 
            0.10, 1.00, 0.40, 0.05,
            help="Ngưỡng độ tin cậy tối thiểu để phát hiện biển số xe."
        )
        speed_mode_label = st.selectbox(
            "Tốc độ nhận diện (OCR Mode)",
            ["Siêu nhanh (Fast)", "Cân bằng (Balanced)", "Chính xác cao (Accurate)"],
            index=1,
            help="Siêu nhanh: OCR 1 lượt. Cân bằng: Tối ưu hoá tốc độ. Chính xác: Chạy ensemble đầy đủ."
        )
        speed_mode = "balanced"
        if "Siêu nhanh" in speed_mode_label:
            speed_mode = "fast"
        elif "Chính xác" in speed_mode_label:
            speed_mode = "accurate"
 
    uploaded_file = st.file_uploader(
        "Upload ảnh biển số",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
    )
 
    if uploaded_file is None:
        st.info("Hãy kéo thả ảnh vào khung trên hoặc bấm **Browse files**.")
        return
 
    image = Image.open(uploaded_file).convert("RGB")
 
    st.success("✅ Tải ảnh lên thành công.")
    st.image(image, caption="Ảnh gốc", use_container_width=True)
 
    st.divider()
 
    if st.button("🔍 Chạy nhận diện", use_container_width=True):
        st.info("Đang nhận diện biển số...")
 
        try:
            import cv2
            import numpy as np
            from pipeline.image_pipeline import ImagePipeline
 
            # PIL/RGB -> OpenCV/BGR
            image_rgb = np.array(image)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
 
            model_path = ROOT / "models" / "yolov8" / "best.pt"
 
            if model_path.is_file():
                pipeline = ImagePipeline(
                    model_path=str(model_path),
                    conf_threshold=conf_threshold,
                    speed_mode=speed_mode,
                )
            else:
                pipeline = ImagePipeline(
                    model_path=None,
                    conf_threshold=conf_threshold,
                    speed_mode=speed_mode,
                )
                if model_path.exists() and model_path.is_dir():
                    st.warning(
                        "`models/yolov8/best.pt` đang là thư mục dataset, "
                        "chưa phải file model YOLO `.pt` thật. Hệ thống sẽ dùng contour fallback."
                    )
                else:
                    st.warning(
                        "Chưa tìm thấy file model YOLO `.pt`. "
                        "Hệ thống sẽ dùng contour fallback."
                    )
 
            result = pipeline.process(
                image=image_bgr,
                source_file=uploaded_file.name,
            )
 
            if result.get("status") == "error":
                st.error(f"Lỗi pipeline: {result.get('error')}")
                return
 
            st.success("✅ Nhận diện hoàn tất")
 
            annotated = result.get("annotated_image")
            detections = result.get("detections", [])
 
            if annotated is not None:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Split layout: Image on the Left, HUD plate on the Right (in the black column space)
                if detections:
                    col_img, col_hud = st.columns([7, 3])
                    with col_img:
                        st.image(
                            annotated_rgb,
                            caption="Ảnh kết quả sau nhận diện",
                            use_container_width=True,
                        )
                    with col_hud:
                        # Sort detections top to bottom by vertical center
                        sorted_dets = sorted(
                            detections,
                            key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2 if d.get("bbox") else 0
                        )
                        
                        hud_cards_html = ""
                        for idx, det in enumerate(sorted_dets, start=1):
                            plate_text = det.get("formatted_text") or det.get("raw_text") or "N/A"
                            is_valid = det.get("is_valid", False)
                            status_label = "OK" if is_valid else "?"
                            status_color = "#00FF66" if is_valid else "#FF3366"
                            
                            hud_cards_html += f"""
                            <div class="hud-panel-right">
                                <div class="hud-header">
                                    <span class="hud-badge">BIỂN SỐ #{idx}</span>
                                    <span class="hud-status" style="color: {status_color}; border-color: {status_color}">{status_label}</span>
                                </div>
                                <div class="hud-plate-text">{plate_text}</div>
                                <div style="font-size: 0.95rem; color: #A3B3CC; margin-top: 12px; font-family: 'Rajdhani', sans-serif; line-height: 1.4; text-align: left;">
                                    Tỉnh/thành: <b style="color: #00D4FF;">{det.get("province", "Unknown")}</b><br/>
                                    Loại biển: <b style="color: #FFD700;">{det.get("plate_type", "Unknown")}</b>
                                </div>
                            </div>
                            """
                            
                        st.markdown(
                            f"""
                            <div class="hud-container-scrollable">
                                {hud_cards_html}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.image(
                        annotated_rgb,
                        caption="Ảnh kết quả sau nhận diện",
                        use_container_width=True,
                    )
 
            if not detections:
                st.warning("Không phát hiện được biển số nào.")
                return
 
        except Exception as e:
            st.error(f"Lỗi khi chạy pipeline: {e}")
        # TODO: Sau khi upload ổn, nối pipeline thật tại đây.
        #
        # Ví dụ nếu bạn có class xử lý ảnh:
        #
        # from pipeline.recognizer import LicensePlateRecognizer
        # recognizer = LicensePlateRecognizer()
        # result = recognizer.process_image(image)
        # st.write(result)
 
 
def render_video_recognition():
    render_header()
 
    st.subheader("🎬 Nhận diện qua Video")
    st.markdown("*Tải video lên để tự động nhận diện tất cả biển số xuất hiện*")
 
    # ── Sidebar settings ──────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**⚙️ Cấu hình Video**")
        frame_skip = st.slider("Tần suất quét (Bỏ qua N khung hình)", 1, 30, 5, 1,
                               help="Quét mỗi N khung hình để tăng tốc độ xử lý.")
        conf_threshold = st.slider("Độ tin cậy phát hiện (Confidence)", 0.10, 1.00, 0.40, 0.05,
                               help="Giảm ngưỡng độ tin cậy nếu biển số nhỏ hoặc ở xa camera.")
        max_frames_inp = st.number_input("Max frames (0 = tất cả)", min_value=0, value=0)
        diff_thr = st.slider("Bỏ frame tĩnh (0=tắt)", 0.0, 5.0, 1.5, 0.1,
                             help="Bỏ qua frame gần như giống hệt frame trước")
        speed_mode_label = st.selectbox(
            "Tốc độ nhận diện (OCR Mode)",
            ["Siêu nhanh (Fast)", "Cân bằng (Balanced)", "Chính xác cao (Accurate)"],
            index=0,  # Default to Fast mode for videos to process quickly
            help="Siêu nhanh: OCR 1 lượt. Cân bằng: Tối ưu hoá tốc độ. Chính xác: Chạy ensemble đầy đủ."
        )
        speed_mode = "balanced"
        if "Siêu nhanh" in speed_mode_label:
            speed_mode = "fast"
        elif "Chính xác" in speed_mode_label:
            speed_mode = "accurate"
 
    # ── Upload ────────────────────────────────────────────────────
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="video_uploader",
    )
 
    if uploaded_video is None:
        st.markdown("""
        <div style='text-align:center;padding:48px;border:2px dashed rgba(0,212,255,0.3);border-radius:16px;'>
            <div style='font-size:2.5rem'>🎬</div>
            <div style='color:#6B7A8D;margin-top:10px'>Upload video MP4 / AVI / MOV / MKV</div>
        </div>
        """, unsafe_allow_html=True)
        return
 
    # ── Save to temp ──────────────────────────────────────────────
    import tempfile, os, time, cv2, pandas as pd
    suffix = f".{uploaded_video.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name
 
    out_path = str(ROOT / "outputs" / "result_videos" / f"result_{uploaded_video.name}")
 
    # Setup columns side-by-side: Video player on Left, HUD cards on Right
    col_video, col_hud = st.columns([6, 4])
 
    # Check if this video has already been processed in this session
    if "processed_videos" not in st.session_state:
        st.session_state["processed_videos"] = {}
 
    video_result = st.session_state["processed_videos"].get(uploaded_video.name)
    run_btn = False
 
    with col_video:
        if video_result and Path(out_path).exists():
            st.video(out_path)
        else:
            st.video(tmp_path)
            
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            run_btn = st.button("▶️ Bắt đầu xử lý", type="primary", use_container_width=True)
        with btn_col2:
            if video_result:
                if st.button("🔄 Chạy lại", type="secondary", use_container_width=True):
                    st.session_state["processed_videos"].pop(uploaded_video.name, None)
                    safe_rerun()
 
    with col_hud:
        if video_result:
            detections = video_result.get("detections", [])
            if detections:
                detections.sort(key=lambda d: (not d.get("is_valid", False), d.get("timestamp_sec", 0)))
                
                hud_cards_html = ""
                for idx, det in enumerate(detections, start=1):
                    plate_text = det.get("formatted_text") or det.get("raw_text") or "N/A"
                    is_valid = det.get("is_valid", False)
                    status_label = "OK" if is_valid else "?"
                    status_color = "#00FF66" if is_valid else "#FF3366"
                    ts = det.get("timestamp_sec", 0)
                    
                    hud_cards_html += f"""
                    <div class="hud-panel-right">
                        <div class="hud-header">
                            <span class="hud-badge">BIỂN SỐ #{idx} ({ts:.1f}s)</span>
                            <span class="hud-status" style="color: {status_color}; border-color: {status_color}">{status_label}</span>
                        </div>
                        <div class="hud-plate-text">{plate_text}</div>
                        <div style="font-size: 0.95rem; color: #A3B3CC; margin-top: 12px; font-family: 'Rajdhani', sans-serif; line-height: 1.4; text-align: left;">
                            Tỉnh/thành: <b style="color: #00D4FF;">{det.get("province", "Unknown")}</b><br/>
                            Loại biển: <b style="color: #FFD700;">{det.get("plate_type", "Unknown")}</b>
                        </div>
                    </div>
                    """
                
                st.markdown(
                    f"""
                    <div class="hud-container-scrollable">
                        {hud_cards_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("ℹ️ Không tìm thấy biển số nào.")
        else:
            st.markdown(
                """
                <div style='text-align:center;padding:48px;border:2px dashed rgba(0,212,255,0.15);border-radius:12px;background:rgba(15,22,41,0.5);'>
                    <div style='font-size:1.8rem;margin-bottom:8px;'>🔍</div>
                    <div style='color:#8892A0;font-size:0.9rem;'>Chưa có kết quả nhận diện.<br>Hãy bấm <b>Bắt đầu xử lý</b> ở cột bên để quét biển số.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
 
    if run_btn:
        with col_video:
            prog_bar   = st.progress(0)
            col_p = st.columns(4)
            lbl_frame  = col_p[0].empty()
            lbl_plates = col_p[1].empty()
            lbl_fps    = col_p[2].empty()
            lbl_eta    = col_p[3].empty()
 
            def on_progress(cur, total, stats):
                pct = min(cur / max(total, 1), 1.0)
                prog_bar.progress(pct)
                lbl_frame.metric("Frame",        f"{cur:,}/{total:,}")
                lbl_plates.metric("Biển tìm được", stats.get("plates_found", 0))
                lbl_fps.metric("Tốc độ",         f"{stats.get('fps_live', 0):.1f} FPS")
                eta = stats.get("eta_sec", 0)
                lbl_eta.metric("Còn lại",        f"{eta}s" if eta > 0 else "—")
 
            from pipeline.video_pipeline import VideoPipeline
            mp = str(ROOT / "models" / "yolov8" / "best.pt")
            pipeline = VideoPipeline(
                model_path=mp if Path(mp).exists() else None,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
                diff_threshold=diff_thr,
                speed_mode=speed_mode,
            )
 
            max_f = max_frames_inp if max_frames_inp > 0 else None
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
 
            with st.spinner("Đang xử lý video..."):
                result = pipeline.process_video(
                    tmp_path,
                    output_path=out_path,
                    progress_callback=on_progress,
                    max_frames=max_f,
                )
 
            prog_bar.progress(1.0)
 
            if result.get("status") == "error":
                st.error(f"❌ Lỗi: {result.get('error')}")
            else:
                st.session_state["processed_videos"][uploaded_video.name] = result
 
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        safe_rerun()
 
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
 
 
def render_history():
    render_header()
 
    st.subheader("📊 Lịch sử & Nhật ký nhận diện")
    st.write("Danh sách các biển số xe đã được hệ thống nhận diện và lưu trữ.")
 
    try:
        from storage.database import PlateDatabase
        import pandas as pd
 
        db = PlateDatabase()
        logs = db.get_logs(limit=100)
 
        if logs:
            df_data = []
            for log in logs:
                df_data.append({
                    "Thời gian": log.get("timestamp", "").replace("T", " ")[:19] if log.get("timestamp") else "N/A",
                    "Nguồn quét": "Ảnh" if log.get("source_type") == "image" else ("Video" if log.get("source_type") == "video" else log.get("source_type")),
                    "Tên tệp tin": log.get("source_file", "N/A"),
                    "Biển số xe": log.get("plate_text", "N/A"),
                    "Độ tin cậy": f"{log.get('confidence', 0.0):.2f}",
                    "Tỉnh thành": log.get("province_name", "Không rõ"),
                    "Loại biển số": log.get("plate_type", "Không rõ"),
                    "Trạng thái": "✅ Hợp lệ" if log.get("is_valid") == 1 else "❌ Không hợp lệ",
                })
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary", use_container_width=True):
                count = db.clear_all_logs()
                st.success(f"✅ Đã xóa toàn bộ {count} bản ghi lịch sử thành công!")
                safe_rerun()
        else:
            st.info("📭 Lịch sử nhận diện hiện đang trống.")
 
    except Exception as e:
        st.warning(f"Lỗi khi tải dữ liệu lịch sử từ database: {e}")
 
 
# =============================================================================
# SIDEBAR
# =============================================================================
 
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Trang chủ"
 
page_options = [
    "Trang chủ",
    "Nhận diện qua Ảnh",
    "Nhận diện qua Video",
    "Lịch sử & Nhật ký",
]
 
with st.sidebar:
    st.title("Menu")
 
    selected_page = st.radio(
        "Chọn chức năng",
        page_options,
        index=page_options.index(st.session_state["current_page"]),
    )
 
    if selected_page != st.session_state["current_page"]:
        st.session_state["current_page"] = selected_page
        safe_rerun()
 
 
# =============================================================================
# ROUTER
# =============================================================================
 
current_page = st.session_state["current_page"]
 
if current_page == "Trang chủ":
    render_home()
elif current_page == "Nhận diện qua Ảnh":
    render_image_recognition()
elif current_page == "Nhận diện qua Video":
    render_video_recognition()
elif current_page == "Lịch sử & Nhật ký":
    render_history()
else:
    render_home()