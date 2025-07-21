import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Deteksi Tumor Otak",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Fungsi untuk Memuat Model (dengan cache agar efisien) ---
@st.cache_resource
def load_model():
    """Memuat model Keras yang sudah dilatih."""
    try:
        model = tf.keras.models.load_model('brain-final.keras') # Ganti jika nama file model Anda berbeda
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# --- Fungsi untuk Prapemrosesan Gambar ---
def preprocess_image(image):
    """
    Melakukan prapemrosesan pada gambar yang diunggah agar sesuai dengan input model.
    """
    # Mengubah gambar PIL ke format array NumPy yang bisa dibaca OpenCV
    image = np.array(image)

    # Konversi dari RGB (PIL) ke BGR (OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ubah ukuran gambar ke 224x224 (sesuai input model EfficientNetB0)
    image = cv2.resize(image, (224, 224))

    # Tambahkan dimensi batch (model mengharapkan input shape: [1, 224, 224, 3])
    image = np.expand_dims(image, axis=0)

    return image

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

    # Prapemrosesan gambar
    processed_image = preprocess_image(image)

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