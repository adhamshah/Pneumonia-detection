"""Model loading and prediction utilities (no Streamlit code here)."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "adham-shah/pediatric-chest-xray-pneumonia-vit"
THRESHOLD = 0.5  # fixed before looking at the test set (see model card)


def load_model(model_id: str = MODEL_ID):
    """Download the fine-tuned model and its processor from the Hugging Face Hub."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model


def predict(image: Image.Image, processor, model) -> dict:
    """Return probabilities for NORMAL / PNEUMONIA for one PIL image."""
    image = image.convert("RGB")  # ViT expects 3 channels
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    p_pneumonia = float(probs[1])
    return {
        "p_normal": float(probs[0]),
        "p_pneumonia": p_pneumonia,
        "label": "PNEUMONIA" if p_pneumonia >= THRESHOLD else "NORMAL",
    }


def looks_like_xray(image: Image.Image) -> tuple[bool, str]:
    """Rough sanity check. A chest X-ray is essentially grayscale.

    This is a heuristic, not a guarantee: it only catches obvious colour photos.
    """
    small = image.convert("RGB").resize((128, 128))
    arr = np.asarray(small).astype(np.float32)
    # Average difference between colour channels; ~0 for grayscale images
    colour_spread = float(np.mean(np.abs(arr[..., 0] - arr[..., 1]) +
                                  np.abs(arr[..., 1] - arr[..., 2])))
    if colour_spread > 12:
        return False, "This image has strong colours, so it is probably not a chest X-ray."
    return True, ""
