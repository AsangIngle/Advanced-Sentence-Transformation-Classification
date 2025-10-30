                                                   Advanced Sentence Transformation Classification
                                                                    Overview

This project develops an advanced NLP model that classifies sentence transformation types such as:

Active ↔ Passive voice
Direct ↔ Indirect speech
Positive ↔ Negative statements

The focus is on accuracy, explainability, and production readiness, demonstrating strong command of transformer-based modeling, interpretability tools (SHAP, LIME, Attention), and practical deployment using Streamlit.


Objective

To design a model that accurately detects the type of grammatical or semantic transformation applied between an original sentence and its transformed version.
Example:
Original: “I am working on a new project.”
Transformed: “I am not working on a new project.”
→ Predicted Transformation: Positive to Negative


Dataset Summary:
Dataset: 3165 samples after deduplication. Stratified 60:20:20 split → Train: 1899, Validation: 633,
Test: 633.



Model Architecture

1.Base Model: DistilBERT (Hugging Face Transformers)
2.Task Type: Sequence Classification
3.Input Format: Concatenated text — “Original: … | Transformed: …”
4.Optimizer: AdamW
5.Learning Rate: 3e-5
6.Epochs: 4
7.Batch Size: 16

Evaluation Metrics
Training (4 Epochs):

Epoch 1 → Val Loss: 0.0103 | Accuracy: 0.9984 | Precision: 0.9984 | Recall: 0.9984 | F1: 0.9984
Epoch 2 → Val Loss: 0.0074 | Accuracy: 0.9968 | Precision: 0.9969 | Recall: 0.9968 | F1: 0.9968
Epoch 3 → Val Loss: 0.0041 | Accuracy: 0.9984 | Precision: 0.9984 | Recall: 0.9984 | F1: 0.9984
Epoch 4 → Val Loss: 0.0043 | Accuracy: 0.9984 | Precision: 0.9984 | Recall: 0.9984 | F1: 0.9984

Test Performance:

Test Loss: 0.0013
Accuracy: 1.000
Precision: 1.000
Recall: 1.000
F1-Score: 1.000

Per-Class Performance:

All six transformation types achieved perfect precision, recall, and F1-scores (1.000 each).

Confusion Matrix Summary:

Each class was perfectly classified with zero misclassifications.
Active ↔ Passive transformations correctly recognized.
Direct ↔ Indirect Speech distinctions learned with complete accuracy.
Sentiment polarity changes (Positive ↔ Negative) also predicted with perfect precision.


Error Analysis:
Minor overlaps observed in early epochs between Direct ↔ Indirect Speech pairs ,likely due to similar syntactic constructions — but resolved after fine-tuning.

Explainability Insights:
To ensure interpretability, three complementary methods were implemented:

1.Attention Visualization:
Highlights tokens that the model attends to most strongly.
Example: high attention to “was”, “by”, “said”, “that”, “not”.

2.SHAP (SHapley Additive exPlanations):
Quantifies the contribution of each word to the prediction probability.
Reveals transformation-triggering tokens (e.g., negations or verb tense markers).

3.LIME (Local Interpretable Model-agnostic Explanations):
Generates localized explanations showing which words push the model toward a specific label.

Error Analysis

1.Misclassifications: None observed on the final test set.
2.Earlier Epoch Behavior:
Minor overlap between Direct ↔ Indirect Speech pairs due to syntactic similarity.
Resolved after fine-tuning and balanced augmentation.


Streamlit Web App
Features

1.Input: Original and Transformed sentences
2.Output: Predicted transformation type + confidence scores

Example Usage

Original: He said, “I am tired.”
Transformed: He said that he was tired.
→ Predicted Transformation: Direct to Indirect Speech


Running the App
  pip install -r requirements.txt
  streamlit run app.py
Ensure that the trained model folder (DistilBERT_Augmented_Predict/) is in the same directory as app.py.


Deployment Readiness

The app can be deployed on:
-Streamlit Cloud (via GitHub repo)
-Hugging Face Spaces (Gradio or Streamlit interface)
-Local hosting using streamlit run


Tools and Libraries
1.Python 3.12
2.PyTorch
3.Transformers (Hugging Face)
4.Scikit-learn
5.Pandas, NumPy
6.SHAP, LIME
7.Matplotlib, Seaborn
8.Streamlit

Key Takeaways
-Built a high-performing and interpretable NLP classifier with DistilBERT.
-Demonstrated explainability using SHAP, LIME, and Attention mechanisms.
-Achieved perfect evaluation metrics through balanced dataset and fine-tuning.
-Delivered a ready-to-deploy Streamlit app for real-time inference.