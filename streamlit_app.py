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

st.set_page_config(layout="wide")
st.title("🧪 Counting Cells Using Convolution")

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def load_tiff(path):
    img = Image.open(path)
    img = np.array(img).astype(np.float32)

    # Normalize safely (TIFFs can have arbitrary ranges)
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


# --------------------------------------------------
# Load 5 TIFF images from labelled folder
# --------------------------------------------------

DATASET_PATH = "labelled"   # change if needed

tiff_files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.tif")))

if len(tiff_files) < 5:
    st.error("Need at least 5 TIFF images in the labelled folder.")
    st.stop()

selected_files = tiff_files[:5]
images = [load_tiff(p) for p in selected_files]
image_names = [os.path.basename(p) for p in selected_files]

selected_idx = st.sidebar.selectbox(
    "Select Image",
    range(5),
    format_func=lambda x: image_names[x]
)

image = images[selected_idx]

# --------------------------------------------------
# SECTION 1 – Human counting
# --------------------------------------------------

st.header("1️⃣ How do humans count cells?")

col1, col2 = st.columns([1, 1])

with col1:
    show_image(image, "Original Microscopy Image")

with col2:
    st.text_input("How many cells do you see?", key="human_count")
    st.text_area(
        "How did you count them? What visual clues did you use?",
        height=150,
        key="human_explanation"
    )

# --------------------------------------------------
# SECTION 2 – Fixed filters
# --------------------------------------------------

st.header("2️⃣ Can a computer count cells using filters?")

filters = {
    "Blob filter (average)": np.ones((3, 3)) / 9,
    "Edge filter": np.array([
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

feature_map = apply_filter(image, kernel)

thresh = st.slider(
    "Threshold",
    float(feature_map.min()),
    float(feature_map.max()),
    float(feature_map.mean())
)

binary, count = threshold_and_count(feature_map, thresh)

c1, c2, c3 = st.columns(3)

with c1:
    show_image(feature_map, "Feature Map (After Convolution)")

with c2:
    show_image(binary, "Thresholded Image")

with c3:
    st.metric("Predicted Cell Count", count)

st.markdown("""
**Think about it:**
- Which filter worked best?
- Did one filter work for all images?
- When did the computer miscount?
""")

# --------------------------------------------------
# SECTION 3 – CNN learns filters
# --------------------------------------------------

st.header("3️⃣ What if the computer learns its own filters?")

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

    # Demo labels (approximate counts, teacher-provided)
    labels = torch.tensor([[15], [18], [12], [20], [16]], dtype=torch.float)

    X = torch.tensor(images).unsqueeze(1).float()

    for _ in range(300):
        preds = model(X)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

model = train_simple_cnn(images)

st.subheader("Learned Filters (First Layer)")

filters = model.conv.weight.data.numpy()

cols = st.columns(4)
for i in range(4):
    f = filters[i, 0]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    fig, ax = plt.subplots()
    ax.imshow(f, cmap="gray")
    ax.axis("off")
    cols[i].pyplot(fig)

with torch.no_grad():
    pred = model(torch.tensor(image).unsqueeze(0).unsqueeze(0).float())
    st.metric("CNN Predicted Cell Count", int(pred.item()))

st.markdown("""
**Reflect:**
- How are these filters different from the ones you chose?
- Why do you think these filters work better?
- Could a human design these easily?
""")
