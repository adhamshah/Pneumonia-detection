"""Streamlit app: pediatric chest X-ray pneumonia screening demo (educational only)."""
import streamlit as st
from PIL import Image

from inference import MODEL_ID, THRESHOLD, load_model, looks_like_xray, predict

st.set_page_config(page_title="Pediatric Chest X-ray Pneumonia Demo", page_icon="🫁", layout="centered")


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def get_model():
    return load_model()


st.title("🫁 Pediatric Chest X-ray Pneumonia Demo")
st.error(
    "**Educational demo. Not a medical device. Not for diagnosis or clinical decisions.** "
    "If you or a child is unwell, contact a doctor."
)

st.markdown(
    "A Vision Transformer (ViT-base), fine-tuned with transfer learning, estimates whether a "
    "**pediatric (age 1-5) frontal chest X-ray** looks NORMAL or like PNEUMONIA."
)

with st.expander("About the model and its limits"):
    st.markdown(
        f"""
- Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}), fine-tuned from `google/vit-base-patch16-224-in21k`.
- Training data: Kermany et al. (2018), one hospital in Guangzhou, children aged 1-5 only.
- Official test set (624 images): AUC 0.987, sensitivity 0.992, **specificity 0.821**
  (about 18% of normal X-rays were wrongly flagged).
- It was **not** tested on adults, other hospitals, other scanners or phone photos of films.
- Images with markers, tubes or wires may confuse it.
"""
    )

uploaded = st.file_uploader("Upload a chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.load()
    except Exception:
        st.error("Could not read this file as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    ok, warning = looks_like_xray(image)
    if not ok:
        st.warning(warning + " The result below is not meaningful.")
    st.caption("Reminder: this model was trained only on children aged 1-5. Results for other ages are unreliable.")

    processor, model = get_model()
    with st.spinner("Analysing..."):
        result = predict(image, processor, model)

    st.subheader("Result")
    if result["label"] == "PNEUMONIA":
        st.warning(f"The model flags this image as **possible pneumonia** (threshold {THRESHOLD}).")
    else:
        st.success(f"The model does **not** flag this image (threshold {THRESHOLD}).")

    c1, c2 = st.columns(2)
    c1.metric("P(pneumonia)", f"{result['p_pneumonia']:.1%}")
    c2.metric("P(normal)", f"{result['p_normal']:.1%}")
    st.progress(result["p_pneumonia"])
    st.caption(
        "Probabilities are model scores, not clinical risk. On the test set the model raised false alarms "
        "on 42 of 234 normal images, so a flag is a reason for review, not a diagnosis."
    )
else:
    st.info("Upload an image to begin.")

st.divider()
st.caption("Data: Kermany et al. (2018), CC BY 4.0. Model: Apache-2.0. Built for a university deep learning project.")
