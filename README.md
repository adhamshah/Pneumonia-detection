# Pediatric Chest X-ray Pneumonia Demo

Streamlit app that loads a fine-tuned Vision Transformer from the Hugging Face Hub
([adham-shah/pediatric-chest-xray-pneumonia-vit](https://huggingface.co/adham-shah/pediatric-chest-xray-pneumonia-vit))
and classifies pediatric chest X-rays as NORMAL or PNEUMONIA.

**Educational project. Not a medical device. Not for diagnosis.**

## Results (official test set, 624 images, threshold 0.5)
| Model | AUC | Sensitivity | Specificity | Accuracy | F1 |
|---|---|---|---|---|---|
| Frozen backbone (baseline) | 0.919 | 0.728 | 0.953 | 0.812 | 0.829 |
| Fine-tuned | 0.987 | 0.992 | 0.821 | 0.928 | 0.945 |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - Streamlit interface
- `inference.py` - model loading, prediction, input sanity check
- `requirements.txt` - dependencies (CPU PyTorch)

## Data and licenses
Kermany et al. (2018), CC BY 4.0. Base model google/vit-base-patch16-224-in21k, Apache-2.0.
Limitations: children aged 1-5, single hospital, many false alarms on normal images.
