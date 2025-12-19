import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Traffic Density Detection",
    page_icon="🚦",
    layout="wide"
)

# ===============================
# HEADER
# ===============================
st.title("Traffic Density Detection")
st.subheader("Analisis Kepadatan Lalu Lintas Berbasis Gambar")

st.markdown("""
Unggah **satu gambar jalan raya**, sistem akan:
1. Mendeteksi kendaraan
2. Menghitung kepadatan berbobot
3. Menentukan status lalu lintas
""")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ===============================
# DENSITY CONFIG
# ===============================
WEIGHT_MOTOR = 1
WEIGHT_CAR = 2

def classify_density(score):
    if score <= 8:
        return "LANCAR", "🟢", "green"
    elif score <= 16:
        return "SEDANG", "🟡", "orange"
    else:
        return "PADAT", "🔴", "red"

# ===============================
# UPLOAD IMAGE
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Jalan Raya",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (640, 360))

    results = model(img, conf=0.25)

    motor_count = 0
    car_count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == "motor":
                motor_count += 1
                color = (0, 255, 255)
            elif class_name == "mobil":
                car_count += 1
                color = (255, 0, 0)
            else:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    density_score = motor_count * WEIGHT_MOTOR + car_count * WEIGHT_CAR
    status, emoji, status_color = classify_density(density_score)

    # ===============================
    # LAYOUT 2 KOLOM
    # ===============================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(img, caption="Hasil Deteksi Kendaraan", use_container_width=True)

    with col2:
        st.markdown("### Hasil Analisis")

        st.metric("🏍️ Motor", motor_count)
        st.metric("🚗 Mobil", car_count)
        st.metric("📈 Skor Kepadatan", density_score)

        st.markdown("---")
        st.markdown(
            f"""
            <h2 style="color:{status_color}; text-align:center;">
            {emoji} {status}
            </h2>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("⬆️ Silakan upload gambar untuk memulai analisis")

