import streamlit as st
import numpy as np
import glob, os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import torchvision.models as models

# ==================================================
# Page setup
# ==================================================

st.set_page_config(layout="wide")
st.title("🧪 Counting Cells Using Convolution")

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
        "3️⃣ Filters learned by a CNN"
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
        st.text_input("How many cells do you see?")
        st.text_area(
            "How did you count them? What visual clues did you use?",
            height=180
        )

# ==================================================
# SECTION 2 — Hand-designed filters
# ==================================================

elif page == "2️⃣ Counting with filters":
    st.header("2️⃣ Can a computer count cells using filters?")

    st.markdown(
        "A computer applies the **same filter everywhere** in the image. "
        "The output becomes strong where the image matches the filter."
    )

    filters = {
        "Blob filter (average)": np.ones((3, 3)) / 9,
        "Edge detector": np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]),
        "Vertical edge": np.array([
            [-1,  2, -1],
            [-1,  2, -1],
            [-1,  2, -1]
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

    st.subheader("From response to counting")

    thresh = st.slider(
        "Threshold",
        float(feature_map.min()),
        float(feature_map.max()),
        float(feature_map.mean())
    )

    binary, count = threshold_and_count(feature_map, thresh)

    c3, c4 = st.columns(2)
    with c3:
        show_image(binary, "Thresholded image")
    with c4:
        st.metric("Predicted count (filter-based)", count)

    st.markdown("""
    **Think about it:**
    - Where does the output become bright?
    - Which filter works best?
    - Why does this sometimes fail?
    """)

# ==================================================
# SECTION 3 — PRETRAINED CNN FILTERS (OPEN SOURCE)
# ==================================================

elif page == "3️⃣ Filters learned by a CNN":
    st.header("3️⃣ Filters learned automatically by a CNN")

    st.markdown(
        "These filters come from an **open-source CNN (ResNet-18)** trained on "
        "millions of images. They were **not designed by humans**.\n\n"
        "The **first layer** of a CNN learns general visual patterns like edges, "
        "blobs, and textures — useful even for microscopy images."
    )

    @st.cache_resource
    def load_pretrained_filters():
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        weights = model.conv1.weight.detach().cpu().numpy()
        return weights.mean(axis=1)  # RGB → grayscale

    kernels = load_pretrained_filters()

    st.subheader("Example learned filters (first CNN layer)")

    cols = st.columns(8)
    for i in range(8):
        k = kernels[i]
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)

        fig, ax = plt.subplots()
        ax.imshow(k, cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)

    st.markdown("""
    **Observe:**
    - These filters were learned, not hand-designed
    - Many filters work together
    - CNNs discover useful patterns automatically
    """)
