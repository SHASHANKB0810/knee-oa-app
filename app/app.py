import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
# 🔧 Environment fixes (keep these at top)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown

# -------------------------------
# 📁 Paths (robust)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

ICON_PATH = os.path.join(BASE_DIR, "img", "mcd.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "..", "src", "models", "model_Xception_ft.hdf5")

# -------------------------------
# 📥 Download model if not exists
# -------------------------------
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = "https://drive.google.com/uc?id=16gSCL2s2t_itsj3a0myjY7kz9WudOBhC"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# 🤖 Load model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------------
# 🔥 Grad-CAM setup
# -------------------------------
grad_model = tf.keras.models.clone_model(model)
grad_model.set_weights(model.get_weights())
grad_model.layers[-1].activation = None

grad_model = tf.keras.models.Model(
    inputs=[grad_model.inputs],
    outputs=[
        grad_model.get_layer("global_average_pooling2d_1").input,
        grad_model.output,
    ],
)

# -------------------------------
# 🖼️ UI Setup
# -------------------------------
icon = Image.open(ICON_PATH)

st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon=icon,
)

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

# -------------------------------
# 🔥 Grad-CAM functions
# -------------------------------
def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


# -------------------------------
# 📊 Sidebar
# -------------------------------
with st.sidebar:
    st.image(icon)
    st.subheader("Knee Osteoarthritis Analysis")
    uploaded_file = st.file_uploader("Upload X-ray image")

# -------------------------------
# 📊 Main UI
# -------------------------------
st.header("Severity Analysis of Arthrosis in the Knee")

col1, col2 = st.columns(2)

if uploaded_file is not None:
    with col1:
        st.subheader("Input Image")
        st.image(uploaded_file, use_column_width=True)

        # Preprocess
        img = Image.open(uploaded_file).convert("RGB").resize(target_size)
        img = np.array(img)

        if st.button("Predict"):
            img_array = np.expand_dims(img, axis=0)
            img_array = np.float32(img_array)
            img_array = tf.keras.applications.xception.preprocess_input(img_array)

            with st.spinner("Predicting..."):
                preds = model.predict(img_array)

            preds = 100 * preds[0]

            index = np.argmax(preds)
            probability = preds[index]

            st.subheader("Prediction")
            st.metric(
                label="Severity Grade",
                value=f"{class_names[index]} ({probability:.2f}%)",
            )

            # Save for second column
            st.session_state["preds"] = preds
            st.session_state["img"] = img
            st.session_state["img_array"] = img_array

# -------------------------------
# 📊 Grad-CAM + Graph
# -------------------------------
if "preds" in st.session_state:
    with col2:
        st.subheader("Explainability")

        heatmap = make_gradcam_heatmap(
            grad_model, st.session_state["img_array"]
        )
        cam_image = save_and_display_gradcam(
            st.session_state["img"], heatmap
        )

        st.image(cam_image, use_column_width=True)

        st.subheader("Analysis")

        preds = st.session_state["preds"]

        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(class_names, preds)

        for i, p in enumerate(preds):
            ax.text(p + 1, i, f"{p:.2f}%")

        ax.set_xlim([0, 100])
        ax.grid(axis="x")

        st.pyplot(fig)
