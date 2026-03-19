import os
import numpy as np
import cv2
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="OralSense - Dental Lesion AI",
    page_icon="🦷",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #0b5394;
    margin-bottom: 0.2rem;
}
.sub-text {
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 1rem;
}
.section-card {
    background-color: #f7f9fc;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
}
.pred-box {
    background-color: #eef6ff;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #d0e3ff;
}
.small-note {
    font-size: 13px;
    color: #666666;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🦷 OralSense — Dental Lesion AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">AI-assisted oral lesion classification with visual explanation using our final model.</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# =========================
# Model definition
# Must match training exactly
# =========================
class CILMP_Lite_NoLeak(nn.Module):
    def __init__(self, num_classes, text_dim=384, r=16):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None)
        self.vision = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vision_dim = 1280

        self.low_rank = nn.Sequential(
            nn.Linear(self.vision_dim, r),
            nn.ReLU(),
            nn.Linear(r, text_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.vision_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, class_text_embeds):
        """
        img: (B,3,224,224)
        class_text_embeds: (C,text_dim)
        """
        B = img.size(0)

        v = self.avgpool(self.vision(img)).view(B, -1)   # (B,1280)
        q = self.low_rank(v)                             # (B,text_dim)

        qn = F.normalize(q, dim=1)
        en = F.normalize(class_text_embeds, dim=1)
        attn_logits = qn @ en.T                          # (B,C)
        w = F.softmax(attn_logits, dim=1)
        prompt = w @ class_text_embeds                   # (B,text_dim)

        fused = torch.cat([v, prompt], dim=1)            # (B, 1280+text_dim)
        logits = self.classifier(fused)
        return logits, attn_logits


# =========================
# Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input_, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x, class_text_embeds, class_idx):
        self.model.zero_grad(set_to_none=True)

        logits, _ = self.model(x, class_text_embeds)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze(1)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()[0]


def overlay_cam_on_image(pil_img, cam, alpha=0.45):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img + alpha * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


# =========================
# Helper text / class descriptions
# Optional: shown in UI for predicted class
# =========================
CLASS_DESCRIPTIONS = {
    "Leukoplakia": "White plaque-like lesion that cannot usually be rubbed off.",
    "Normal Mucosa": "Healthy oral mucosa with normal pink smooth appearance.",
    "Squamous cell carcinoma": "Potential malignant lesion with ulceration or irregular tissue appearance.",
    "Apthous Ulcer": "Painful ulcerative lesion, often shallow with inflammatory halo.",
    "osmf": "Oral submucous fibrosis with blanched mucosa and restricted movement.",
    "Lichen planus": "Lesion showing reticular or lacy white streak patterns."
}


# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Model Settings")

default_ckpt = "cilmp_dental_v3.pt"
#ckpt_path = st.sidebar.text_input("Checkpoint path", value=default_ckpt)

show_gradcam = st.sidebar.checkbox("Enable Grad-CAM", value=True)
topk = st.sidebar.slider("Top-K predictions", min_value=2, max_value=5, value=3)
cam_alpha = st.sidebar.slider("Grad-CAM overlay strength", min_value=0.20, max_value=0.80, value=0.45)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="small-note">Tip: Keep the checkpoint file in the same project folder or provide the full path.</div>',
    unsafe_allow_html=True
)


# =========================
# Transforms
# =========================
val_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================
# Load model
# =========================
@st.cache_resource
def load_model_and_assets(path):
    device = "cpu"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)

    classes = ckpt["classes"]
    text_dim = ckpt.get("text_dim", 384)
    r = ckpt.get("r", 16)

    model = CILMP_Lite_NoLeak(
        num_classes=len(classes),
        text_dim=text_dim,
        r=r
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_text_embeds = ckpt["class_text_embeds"].to(device).float()

    return model, class_text_embeds, classes, device, ckpt


try:
    model, class_text_embeds, classes, device, ckpt = load_model_and_assets(ckpt_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# =========================
# Upload section
# =========================
st.subheader("Upload Image")
uploaded = st.file_uploader(
    "Upload an oral lesion image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    try:
        pil_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        st.stop()

    x = val_tfms(pil_img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        logits, attn_logits = model(x, class_text_embeds)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_name = classes[pred_idx]
    pred_prob = float(probs[pred_idx])

    # Grad-CAM
    overlay = None
    if show_gradcam:
        try:
            cam_engine = GradCAM(model, model.vision[-1])
            cam = cam_engine(x, class_text_embeds, pred_idx)
            cam_engine.remove()
            overlay = overlay_cam_on_image(pil_img, cam, alpha=cam_alpha)
        except Exception as e:
            st.warning(f"Grad-CAM could not be generated: {e}")

    # =========================
    # Layout
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_img, caption="Input Image", use_container_width=True)

    with col2:
        if overlay is not None:
            st.image(overlay, caption="Model Attention (Grad-CAM)", use_container_width=True)
        else:
            st.info("Grad-CAM not available.")

    st.markdown("---")

    # prediction metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", pred_name)
    c2.metric("Confidence", f"{pred_prob:.4f}")
    c3.metric("Total Classes", f"{len(classes)}")

    st.markdown("### Predicted Class Summary")
    st.markdown(
        f"""
<div class="pred-box">
<b>Predicted Lesion:</b> {pred_name}<br>
<b>Confidence:</b> {pred_prob:.4f}<br><br>
<b>Clinical Hint:</b> {CLASS_DESCRIPTIONS.get(pred_name, "No description available.")}
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # top-k predictions
    st.subheader("Top Predictions")
    top_idxs = np.argsort(-probs)[:topk]

    for idx in top_idxs:
        idx = int(idx)
        label = classes[idx]
        prob = float(probs[idx])
        st.progress(prob, text=f"{label} — {prob:.4f}")

    st.markdown("---")

    # raw probability table
    st.subheader("Class Probabilities")
    prob_data = []
    for i, cls_name in enumerate(classes):
        prob_data.append({
            "Class": cls_name,
            "Probability": float(probs[i])
        })

    prob_data = sorted(prob_data, key=lambda x: x["Probability"], reverse=True)
    st.dataframe(prob_data, use_container_width=True)

    # optional attention over class prompts
    st.subheader("Prototype Attention Scores")
    attn_scores = torch.softmax(attn_logits, dim=1).cpu().numpy()[0]

    attn_data = []
    for i, cls_name in enumerate(classes):
        attn_data.append({
            "Class Prototype": cls_name,
            "Attention Weight": float(attn_scores[i])
        })

    attn_data = sorted(attn_data, key=lambda x: x["Attention Weight"], reverse=True)
    st.dataframe(attn_data, use_container_width=True)

else:
    st.info("Upload an image to run inference.")
