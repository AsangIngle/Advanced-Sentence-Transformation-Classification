import streamlit as st
import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_DIR="C:/Immverse/DistilBERT_Augmented-20251029T181902Z-1-001/DistilBERT_Augmented"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model.eval()

labels = list(model.config.id2label.values())

st.title("Sentence Transformation Classifier")

orig = st.text_area("Original Sentence", "He said, 'I am tired.'")
trans = st.text_area("Transformed Sentence", "He said that he was tired.")
btn = st.button("Classify Transformation")

if btn:
    text = f"Original: {orig} | Transformed: {trans}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]

    st.subheader(f"Predicted Transformation: {pred_label}")
    st.progress(float(probs[pred_idx]))
    st.write("Confidence Scores:")
    for label, p in zip(labels, probs):
        st.write(f"{label}: {p:.3f}")