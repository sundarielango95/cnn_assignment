import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

# ==================================================
# Page setup
# ==================================================

st.set_page_config(layout="wide")
st.title("🧪 Counting Cells Using Convolution")

# ==================================================
# Utility functions
# ==================================================

def load_tiff(path):
    """Load TIFF robustly and return normalized 2D grayscale image."""
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32)
    img = (img - img.min()) / (img.max() + 1e-8)
    return img


def apply_filter(image, kernel):
    return ndimage.convolve(image, kernel, mode="reflect")


def threshold_and_count(feature_map, thresh):
    binary = feature_map > thresh
    labeled, num = ndimage.label(binary)
    return binary, num


def show_image(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)

# ==================================================
# Load images
# ==================================================

DATASET_PATH = "labelled"
tiff_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.tif*")))

if len(tiff_files) < 5:
    st.error("Need at least 5 TIFF images in the labelled folder.")
    st.stop()

files = tiff_files[:5]
images = [load_tiff(p) for p in files]
names = [os.path.basename(p) for p in files]

idx = st.sidebar.selectbox(
    "Select image",
    range(5),
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
        "3️⃣ CNN learns filters"
    ]
)

# ==================================================
# SECTION 1 — Human counting
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
# SECTION 2 — BEFORE / AFTER CONVOLUTION
# ==================================================

elif page == "2️⃣ Counting with filters":
    st.header("2️⃣ Can a computer count cells using filters?")

    st.markdown(
        "A computer applies the **same filter everywhere** in the image. "
        "The output becomes bright where the image matches the filter."
    )

    # -------------------------------
    # Filter selection
    # -------------------------------

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

    # -------------------------------
    # Apply convolution
    # -------------------------------

    feature_map = apply_filter(image, kernel)

    # -------------------------------
    # Visualise before / after
    # -------------------------------

    st.subheader("🔹 Effect of convolution")

    c1, c2 = st.columns(2)

    with c1:
        show_image(image, "Original image")

    with c2:
        show_image(feature_map, "After convolution (feature map)")

    # -------------------------------
    # Thresholding + counting
    # -------------------------------

    st.subheader("🔹 From response to counting")

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
        st.metric("Predicted cell count", count)

    st.markdown("""
    **Think about it:**
    - Where does the output become bright?
    - Does that correspond to cells?
    - Which filter works best, and why?
    """)

# ==================================================
# SECTION 3 — CNN learns filters
# ==================================================

elif page == "3️⃣ CNN learns filters":
    st.header("3️⃣ What if the computer learns its own filters?")

    if "model" not in st.session_state:
        st.session_state.model = None

    @st.cache_resource
    def train_cnn(images):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, 1)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                return self.fc(x.view(x.size(0), -1))

        model = Net()
        X = torch.tensor(images).unsqueeze(1)
        y = torch.tensor([[15], [18], [12], [20], [16]], dtype=torch.float)

        opt = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for _ in range(300):
            opt.zero_grad()
            loss_fn(model(X), y).backward()
            opt.step()

        return model

    if st.button("🚀 Train CNN"):
        with st.spinner("Training CNN..."):
            st.session_state.model = train_cnn(images)

    if st.session_state.model:
        st.subheader("Learned filters (first convolution layer)")
        W = st.session_state.model.conv.weight.detach().numpy()

        cols = st.columns(4)
        for i in range(4):
            f = W[i, 0]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            fig, ax = plt.subplots()
            ax.imshow(f, cmap="gray")
            ax.axis("off")
            cols[i].pyplot(fig)

        with torch.no_grad():
            pred = st.session_state.model(
                torch.tensor(image).unsqueeze(0).unsqueeze(0)
            )
            st.metric("CNN predicted cell count", int(pred.item()))
    else:
        st.info("Press **Train CNN** to let the computer learn its own filters.")
