import streamlit as st
import numpy as np
import glob, os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torchvision.models as models
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import io

# ==================================================
# Page setup
# ==================================================

st.set_page_config(layout="wide")
st.title("🧪 Counting Cells Using Convolution")

# ==================================================
# Initialize session state
# ==================================================

for key in [
    "student_name",
    "s1_q1", "s1_q2",
    "s2_q1", "s2_q2", "s2_q3",
    "s3_q1", "s3_q2"
]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ==================================================
# Utility functions
# ==================================================

def load_tiff(path):
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32)
    img = (img - img.min()) / (img.max() + 1e-8)
    return img


def apply_filter(image, kernel):
    return ndimage.convolve(image, kernel, mode="reflect")


def threshold_and_count(feature_map, thresh):
    binary = feature_map > thresh
    _, num = ndimage.label(binary)
    return binary, num


def show_image(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)

# ==================================================
# Load microscopy images (labelled/)
# ==================================================

IMG_DIR = "labelled"
tiff_files = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.tif")) +
    glob.glob(os.path.join(IMG_DIR, "*.tiff"))
)

if len(tiff_files) == 0:
    st.error("No TIFF images found in labelled/")
    st.stop()

images = [load_tiff(p) for p in tiff_files]
names = [os.path.basename(p) for p in tiff_files]

idx = st.sidebar.selectbox(
    "Select image",
    range(len(images)),
    format_func=lambda i: names[i]
)
image = images[idx]

# ==================================================
# Sidebar navigation
# ==================================================

page = st.sidebar.radio(
    "Go to section",
    [
        "1️⃣ Human cell counting",
        "2️⃣ Counting with filters",
        "3️⃣ Filters learned by a CNN",
        "📄 Download answers"
    ]
)

# ==================================================
# SECTION 1 — Human intuition
# ==================================================

if page == "1️⃣ Human cell counting":
    st.header("1️⃣ How do humans count cells?")

    c1, c2 = st.columns(2)

    with c1:
        show_image(image, "Microscopy image")

    with c2:
        st.session_state.s1_q1 = st.text_input(
            "How many cells do you see?",
            value=st.session_state.s1_q1
        )

        st.session_state.s1_q2 = st.text_area(
            "How did you count the cells? What visual cues did you use?",
            value=st.session_state.s1_q2,
            height=180
        )

# ==================================================
# SECTION 2 — Hand-designed filters
# ==================================================

elif page == "2️⃣ Counting with filters":
    st.header("2️⃣ Counting with hand-designed filters")

    # filters = {
    #     "Blob filter (average)": np.ones((3, 3)) / 9,
    #     "Edge detector": np.array([
    #         [-1, -1, -1],
    #         [-1,  8, -1],
    #         [-1, -1, -1]
    #     ])
    # }
    filters = {
        "Blob filter (average)": np.ones((3, 3)) / 9,
    
        "Gaussian blur": np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 16,
    
        "Edge detector (Laplacian)": np.array([
            [ 0,  1,  0],
            [ 1, -4,  1],
            [ 0,  1,  0]
        ]),
    
        "Vertical edge detector": np.array([
            [-1,  2, -1],
            [-1,  2, -1],
            [-1,  2, -1]
        ]),
    
        "Horizontal edge detector": np.array([
            [-1, -1, -1],
            [ 2,  2,  2],
            [-1, -1, -1]
        ]),
    
        "Sharpening filter": np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
    }


    fname = st.selectbox("Choose a filter", list(filters.keys()))
    kernel = filters[fname]

    feature_map = apply_filter(image, kernel)

    c1, c2 = st.columns(2)
    with c1:
        show_image(image, "Original image")
    with c2:
        show_image(feature_map, "After convolution")

    thresh = st.slider(
        "Threshold",
        float(feature_map.min()),
        float(feature_map.max()),
        float(feature_map.mean())
    )

    binary, count = threshold_and_count(feature_map, thresh)

    show_image(binary, "Thresholded output")
    st.metric("Predicted count (filter-based)", count)

    st.subheader("Think about it")

    st.session_state.s2_q1 = st.text_area(
        "Where does the output become bright?",
        value=st.session_state.s2_q1
    )

    st.session_state.s2_q2 = st.text_area(
        "Which filter works best?",
        value=st.session_state.s2_q2
    )

    st.session_state.s2_q3 = st.text_area(
        "Why does this sometimes fail?",
        value=st.session_state.s2_q3
    )

# ==================================================
# SECTION 3 — Pretrained CNN filters
# ==================================================

elif page == "3️⃣ Filters learned by a CNN":
    st.header("3️⃣ Filters learned automatically by a CNN")

    st.markdown(
        "These filters come from an **open-source CNN (ResNet-18)** trained on "
        "millions of images. They were **not designed by humans**."
    )

    @st.cache_resource
    def load_pretrained_filters():
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        return model.conv1.weight.detach().cpu().numpy().mean(axis=1)

    kernels = load_pretrained_filters()

    cols = st.columns(8)
    for i in range(8):
        k = kernels[i]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        fig, ax = plt.subplots()
        ax.imshow(k, cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)

    st.subheader("Reflect")

    st.session_state.s3_q1 = st.text_area(
        "How are these filters different from hand-designed filters?",
        value=st.session_state.s3_q1
    )

    st.session_state.s3_q2 = st.text_area(
        "Why might learned filters work better?",
        value=st.session_state.s3_q2
    )

# ==================================================
# SECTION 4 — PDF export
# ==================================================

elif page == "📄 Download answers":
    st.header("Download your answers")

    st.session_state.student_name = st.text_input(
        "Student name",
        value=st.session_state.student_name
    )

    if st.button("Generate PDF"):
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = [height - 40]

        def write(text):
            for line in text.split("\n"):
                pdf.drawString(40, y[0], line)
                y[0] -= 14
                if y[0] < 40:
                    pdf.showPage()
                    y[0] = height - 40

        write(f"Student: {st.session_state.student_name}")
        write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        write("\nSection 1: Human cell counting")
        write(f"How many cells did you see? {st.session_state.s1_q1}")
        write(f"How did you count the cells? {st.session_state.s1_q2}")

        write("\nSection 2: Counting with filters")
        write(f"Where does the output become bright? {st.session_state.s2_q1}")
        write(f"Which filter works best? {st.session_state.s2_q2}")
        write(f"Why does this sometimes fail? {st.session_state.s2_q3}")

        write("\nSection 3: CNN filters")
        write(f"How are learned filters different? {st.session_state.s3_q1}")
        write(f"Why might learned filters work better? {st.session_state.s3_q2}")

        pdf.save()
        buffer.seek(0)

        st.download_button(
            "Download PDF",
            buffer,
            file_name="cell_counting_reflections.pdf",
            mime="application/pdf"
        )
