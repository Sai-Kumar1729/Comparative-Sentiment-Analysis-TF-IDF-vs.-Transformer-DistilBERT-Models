Advanced Sentiment Analysis of Mobile Phone Reviews
Author: [Your Name]
Date: September 4, 2025

1. Project Overview
This project performs sentiment analysis on a dataset of over 14,000 customer reviews for the Lenovo K8 Note mobile phone. The primary objective is to classify reviews as either positive or negative.

To demonstrate a comprehensive understanding of Natural Language Processing (NLP) techniques, this project implements and compares two distinct approaches:

A Classic Machine Learning Pipeline: Utilizes traditional text preprocessing and a Bag-of-Words model (TF-IDF) with Logistic Regression.

A State-of-the-Art Transformer Model: Employs a pre-trained DistilBERT model and fine-tunes it on the specific review dataset for a more nuanced, context-aware analysis.

2. Final Results: A Comparative Analysis
The results clearly demonstrate the superior performance of the Transformer-based approach, which leverages contextual understanding of language to achieve a higher accuracy.

Model

Accuracy

Key Technology

TF-IDF + Logistic Regression

~83%

Bag-of-Words

Fine-Tuned Transformer (DistilBERT)

~89%

Attention Mechanism

The ~6% increase in accuracy highlights the advantage of modern NLP architectures in grasping complex linguistic patterns that are often missed by traditional methods.

3. Methodologies
Approach 1: Classic NLP with Bag-of-Words
This baseline model follows a standard, robust machine learning pipeline for text classification.

Text Preprocessing: The raw text is cleaned through a series of steps:

Punctuation and special character removal.

Tokenization (splitting text into individual words).

Stop word removal (e.g., "the", "a", "is").

Lemmatization (reducing words to their root form, e.g., "running" -> "run").

Feature Extraction: The cleaned text is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams (1,2), which captures both single words and two-word phrases.

Model Training: A Logistic Regression classifier is trained on the TF-IDF vectors to learn the relationship between word patterns and sentiment.

Approach 2: Fine-Tuning a Transformer Model
This advanced approach uses a modern deep learning architecture to understand the context and meaning behind the words.

Model: We use distilbert-base-uncased, a smaller, faster version of Google's BERT model. It has been pre-trained on a massive corpus of text, giving it a powerful, general-purpose understanding of the English language.

Tokenization: The raw text is processed by a specific DistilBERT tokenizer, which converts words into a numerical format the model understands. Manual cleaning like stop word removal is not required, as the model learns context from the full sentence.

Fine-Tuning (Transfer Learning): The pre-trained model is then trained for a few more epochs on our specific mobile review dataset. During this process, the model adapts its general language knowledge to become a specialist in predicting sentiment for mobile phone reviews. This training was performed on Google Colab to leverage GPU acceleration.

4. Project Structure
The project is organized into the following directories and files:

sentiment-analysis-project/
│
├── notebooks/
│   ├── 1_BoW_and_Logistic_Regression.ipynb   # Baseline model
│   └── 2_Transformer_Fine_Tuning.ipynb         # Advanced model
│
├── data/
│   └── K8 Reviews.csv                        # The raw dataset
│
├── saved_models/
│   ├── classifier.pickle                     # Saved Logistic Regression model
│   └── tfidfmodel.pickle                     # Saved TF-IDF vectorizer
│
└── README.md                                 # This file

5. How to View This Project
Start with the Baseline: Open notebooks/1_BoW_and_Logistic_Regression.ipynb to review the classic machine learning approach. This notebook can be run on a standard local machine.

Review the Advanced Model: Open notebooks/2_Transformer_Fine_Tuning.ipynb.

Important Note: This notebook requires a GPU for timely execution (30 minutes on Google Colab vs. an estimated 30+ hours on a standard CPU).

The cell outputs, including training logs, performance plots, and the final 89% accuracy score, have been pre-saved. You can view the complete results without needing to re-run the computationally intensive training cells.

6. Conclusion
This project successfully demonstrates the evolution of NLP techniques for sentiment analysis. While the classic Bag-of-Words approach provides a solid baseline, the fine-tuned Transformer model achieves significantly higher accuracy. This is primarily due to its attention mechanism, which allows it to weigh the importance of different words in a sentence and understand context, nuance, and negation far more effectively.