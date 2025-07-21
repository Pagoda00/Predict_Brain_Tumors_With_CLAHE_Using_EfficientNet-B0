import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Tumor Otak",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Fungsi untuk Memuat Model (dengan cache dan unduhan) ---
@st.cache_resource
def load_model():
    """
    Memeriksa, mengunduh jika perlu, dan memuat model Keras yang sudah dilatih.
    """
    model_path = 'brain-final.keras'

    # Cek jika file model belum ada di direktori
    if not os.path.exists(model_path):
        st.info("Model tidak ditemukan. Mengunduh dari Google Drive... (mungkin butuh beberapa saat)")

        # PENTING: Ganti dengan ID file Google Drive Anda
        # Contoh link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        # ID file adalah bagian yang ada di antara /d/ dan /view
        file_id = "GANTI_DENGAN_ID_FILE_GOOGLE_DRIVE_ANDA"

        try:
            gdown.download(id=file_id, output=model_path, quiet=False)
            st.success("Model berhasil diunduh!")
        except Exception as e:
            st.error(f"Gagal mengunduh model: {e}")
            return None

    # Muat model setelah ada
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model dari file lokal: {e}")
        return None

# --- Fungsi untuk Prapemrosesan Gambar ---
def resize_with_padding(image, target_size=224):
    """
    Mengubah ukuran gambar dengan padding untuk menjaga rasio aspek.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size, target_size

    # Tentukan skala dan ukuran baru
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h))

    # Buat kanvas hitam dan letakkan gambar di tengah
    padded_image = np.zeros((target_h, target_w), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded_image[top:top + new_h, left:left + new_w] = resized_image

    return padded_image

def preprocess_image_improved(image_array, image_size=224):
    """
    Pipeline pre-processing yang disempurnakan.
    Input adalah NumPy array.
    """
    # Step 1: Grayscale Conversion
    if len(image_array.shape) > 2 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_array

    # Step 2: Resizing with Padding
    padded_image = resize_with_padding(gray_image, image_size)

    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(padded_image)

    # Step 4: Denoising
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 5: Unsharp Mask
    blurred = cv2.GaussianBlur(denoised_image, (5, 5), 0)
    unsharp_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)

    # Step 6: Konversi ke 3 Channel
    three_channel_image = cv2.cvtColor(unsharp_image, cv2.COLOR_GRAY2RGB)

    # Step 7: Tambahkan dimensi batch
    final_image = np.expand_dims(three_channel_image, axis=0)

    return final_image

# --- Aplikasi Utama Streamlit ---

# Muat model
model = load_model()

# Tampilkan judul dan deskripsi
st.title("🧠 Deteksi Tumor Otak Berbasis MRI")
st.write(
    "Aplikasi ini menggunakan model Deep Learning (EfficientNet-B0) untuk "
    "memprediksi jenis tumor otak dari gambar MRI. Unggah gambar MRI untuk memulai."
)

# Definisikan label kelas sesuai urutan pada saat training
class_labels = ['Glioma', 'Meningioma', 'Tanpa Tumor', 'Pituitary']

# Komponen untuk mengunggah file gambar
uploaded_file = st.file_uploader(
    "Pilih gambar MRI...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
    st.write("")
    st.write("Memproses dan memprediksi...")

    # Konversi gambar PIL ke NumPy array sebelum diproses
    image_np = np.array(image)

    # Prapemrosesan gambar
    processed_image = preprocess_image_improved(image_np)

    # Lakukan prediksi
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = np.max(prediction) * 100

    # Tampilkan hasil prediksi
    st.success(f"**Hasil Prediksi:** {predicted_class_label}")
    st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")

    # Tambahan: Tampilkan probabilitas untuk setiap kelas
    st.write("Probabilitas untuk setiap kelas:")
    for i, label in enumerate(class_labels):
        st.write(f"- {label}: {prediction[0][i]*100:.2f}%")

