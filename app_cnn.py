import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Sprint-2 CNN Image Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===============================
# DARK THEME (FORCED)
# ===============================
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e5e7eb;
}
.block-container {
    padding-top: 2rem;
    max-width: 850px;
}
.card {
    background: rgba(15, 23, 42, 0.85);
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 0 30px rgba(59,130,246,0.25);
    margin-bottom: 1.5rem;
}
.center { text-align: center; }
label, p, span { color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)

# ===============================
# DEVICE
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===============================
# CNN MODEL
# ===============================
class ImageClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ===============================
# ASPECT RATIO PADDING
# ===============================
class ResizeWithPadding:
    def __call__(self, img):
        w, h = img.size
        m = max(w, h)
        pl = (m - w) // 2
        pr = m - w - pl
        pt = (m - h) // 2
        pb = m - h - pt
        return F.pad(img, (pl, pt, pr, pb), fill=0)

# ===============================
# TRANSFORM
# ===============================
infer_transform = transforms.Compose([
    ResizeWithPadding(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ===============================
# LOAD MODELS
# ===============================
seat_model = ImageClassifierCNN(2).to(device)
seat_model.load_state_dict(torch.load("seat_no_aug_final.pth", map_location=device))
seat_model.eval()

toilet_model = ImageClassifierCNN(3).to(device)
toilet_model.load_state_dict(torch.load("toilet_no_aug_final.pth", map_location=device))
toilet_model.eval()

seat_classes = ["Elongated", "Round"]
toilet_classes = ["Aquapro", "Montecarlo", "Smart"]

# ===============================
# PREDICT FUNCTION
# ===============================
def predict(img, model, classes):
    img = infer_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}
    pred_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[pred_class]

    return pred_class, confidence, probabilities

# ===============================
# UI
# ===============================
st.markdown("""
<div class="card center">
    <h1>🚽 Sprint-2 CNN Image Classifier</h1>
    <p>Streamlit + API + CNN </p>
</div>
""", unsafe_allow_html=True)

dataset = st.selectbox("Choose Dataset", ["Seat Cover", "Toilet"])
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300, caption="Uploaded Image")

    if st.button("Predict"):
        if dataset == "Seat Cover":
            prediction, confidence, probabilities = predict(img, seat_model, seat_classes)
        else:
            prediction, confidence, probabilities = predict(img, toilet_model, toilet_classes)

        st.markdown("---")
        st.subheader(f" Prediction: **{prediction}**")

        conf_pct = round(confidence * 100, 2)
        st.markdown(f" Confidence: **{conf_pct}%**")
        st.progress(confidence)

        st.markdown("---")
        st.subheader(" Class Probabilities")

        for cls, prob in probabilities.items():
            pct = round(prob * 100, 2)
            if cls == prediction:
                st.markdown(f"** {cls}: {pct}%**")
            else:
                st.markdown(f"{cls}: {pct}%")
            st.progress(prob)

        st.markdown("---")
        st.subheader(" Test Set Metrics")

        if dataset == "Toilet":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "0.81")
            c2.metric("Precision", "0.80")
            c3.metric("Recall", "0.81")
            c4.metric("F1", "0.80")

        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", "0.59")
            c2.metric("Precision", "0.59")
            c3.metric("Recall", "0.59")
            c4.metric("F1", "0.55")
