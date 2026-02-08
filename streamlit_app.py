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
    """
    Load TIFF robustly and return a normalized 2D grayscale image.
    Handles RGB TIFFs, (H,W,1), etc.
    """
    img = Image.open(path).convert("L")
    img = np.array(img).astype(np.float32)

    img = img - img.min()
    img = img / (img.max() + 1e-8)

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

selected_files = tiff_files[:5]
images = [load_tiff(p) for p in selected_files]
image_names = [os.path.basename(p) for p in selected_files]

selected_idx = st.sidebar.selectbox(
    "Select image",
    range(5),
    format_func=lambda x: image_names[x]
)

image = images[selected_idx]

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

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(image, "Microscopy image")

    with col2:
        st.text_input("How many cells do you see?", key="human_count")
        st.text_area(
            "How did you count them? What visual clues did you use?",
            height=180,
            key="human_explanation"
        )

# ==================================================
# SECTION 2 — Filters + correct visualisation
# ==================================================

elif page == "2️⃣ Counting with filters":
    st.header("2️⃣ Can a computer count cells using filters?")

    st.markdown(
        "A computer does not know what a cell is. "
        "It slides a **filter (kernel)** over the image and responds strongly "
        "where the image matches the filter."
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

    filter_name = st.selectbox("Choose a filter", list(filters.keys()))
    kernel = filters[filter_name]

    # -------------------------------
    # Visualise the filter correctly
    # -------------------------------

    st.subheader("🔹 The filter (kernel)")

    # Normalise ONLY for display
    k = kernel.copy()
    k_min, k_max = k.min(), k.max()

    if k_max > k_min:
        k_vis = (k - k_min) / (k_max - k_min)
    else:
        # Flat kernel (e.g. averaging filter)
        k_vis = np.ones_like(k) * 0.5

    fig, ax = plt.subplots()
    ax.imshow(k_vis, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Filter values (visualised)")
    ax.set_xticks(range(k.shape[1]))
    ax.set_yticks(range(k.shape[0]))
    ax.set_xticklabels(range(k.shape[1]))
    ax.set_yticklabels(range(k.shape[0]))

    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            ax.text(
                j, i,
                f"{k[i, j]:.2f}",
                ha="center",
                va="center",
                color="red",
                fontsize=12
            )

    st.pyplot(fig)

    st.markdown(
        "This filter is slid across the image. "
        "Where the image looks similar to this pattern, "
        "the output becomes bright."
    )

    # -------------------------------
    # Apply convolution
    # -------------------------------

    feature_map = apply_filter(image, kernel)

    thresh = st.slider(
        "Threshold",
        float(feature_map.min()),
        float(feature_map.max()),
        float(feature_map.mean())
    )

    binary, count = threshold_and_count(feature_map, thresh)

    # -------------------------------
    # Before / after comparison
    # -------------------------------

    st.subheader("🔹 Effect of applying the filter")

    c1, c2, c3 = st.columns(3)

    with c1:
        show_image(image, "Original image")

    with c2:
        show_image(feature_map, "After convolution (feature map)")

    with c3:
        show_image(binary, "After thresholding")

    st.metric("Predicted cell count", count)

    st.markdown("""
    **Think about it:**
    - Where does the feature map become bright?
    - Does that match where the cells are?
    - Which filter works best, and why?
    """)

# ==================================================
# SECTION 3 — CNN learns filters (button-triggered)
# ==================================================

elif page == "3️⃣ CNN learns filters":
    st.header("3️⃣ What if the computer learns its own filters?")

    if "cnn_model" not in st.session_state:
        st.session_state.cnn_model = None

    @st.cache_resource
    def train_simple_cnn(images):
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 1)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Teacher-provided approximate counts (demo only)
        labels = torch.tensor([[15], [18], [12], [20], [16]], dtype=torch.float)
        X = torch.tensor(images).unsqueeze(1).float()

        for _ in range(300):
            preds = model(X)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    if st.button("🚀 Train CNN"):
        with st.spinner("Training CNN..."):
            st.session_state.cnn_model = train_simple_cnn(images)

    if st.session_state.cnn_model is not None:
        model = st.session_state.cnn_model

        st.subheader("Learned filters (first convolution layer)")
        learned_filters = model.conv.weight.data.numpy()

        cols = st.columns(4)
        for i in range(4):
            f = learned_filters[i, 0]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            fig, ax = plt.subplots()
            ax.imshow(f, cmap="gray")
            ax.axis("off")
            cols[i].pyplot(fig)

        with torch.no_grad():
            pred = model(
                torch.tensor(image)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            )
            st.metric("CNN predicted cell count", int(pred.item()))
    else:
        st.info("Press **Train CNN** to let the computer learn its own filters.")
