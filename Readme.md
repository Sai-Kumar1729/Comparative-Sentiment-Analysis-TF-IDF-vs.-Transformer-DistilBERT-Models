# 🚀 Comparative Sentiment Analysis: TF-IDF vs. Transformer Models
A comprehensive analysis of mobile phone reviews using classic machine learning and modern deep learning techniques.

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)  
2. [Final Results & Analysis](#2-final-results-a-comparative-analysis)  
3. [Methodologies](#3-methodologies)  
   - [Approach 1: Classic NLP (Bag-of-Words)](#approach-1-classic-nlp-with-bag-of-words)  
   - [Approach 2: Transformer Model (DistilBERT)](#approach-2-fine-tuning-a-transformer-model)  
4. [Project Structure](#4-project-structure)  
5. [How to View This Project](#5-how-to-view-this-project)  
6. [Conclusion](#6-conclusion)  

---

## 1. Project Overview
This project performs sentiment analysis on a dataset of over **14,000 customer reviews** for the Lenovo K8 Note mobile phone.  
The primary objective is to classify reviews as either **positive** or **negative**.

To demonstrate a robust understanding of **Natural Language Processing (NLP)**, this project implements and compares two distinct approaches:

- **Classic Machine Learning Pipeline**: Utilizes a Bag-of-Words model (TF-IDF) with Logistic Regression.  
- **State-of-the-Art Transformer Model**: Employs a pre-trained DistilBERT model, fine-tuned for this specific task.  

---

## 2. Final Results: A Comparative Analysis
The results clearly demonstrate the **superior performance** of the Transformer-based approach, which leverages deep contextual understanding of language.

| Model                           | Accuracy | Key Technology   |
|--------------------------------|----------|-----------------|
| TF-IDF + Logistic Regression   | ~83%     | Bag-of-Words    |
| Fine-Tuned Transformer (DistilBERT) | ~89%     | Attention Mechanism |

📈 The ~6% increase in accuracy highlights the advantage of **modern NLP architectures** in capturing complex linguistic patterns such as sarcasm and negation.

---

## 3. Methodologies  

### Approach 1: Classic NLP with Bag-of-Words
This baseline model follows a standard, robust machine learning pipeline for text classification.

- **Text Preprocessing**:  
  - Remove punctuation and special characters  
  - Tokenization (split text into words)  
  - Stop word removal (e.g., *the, a, is*)  
  - Lemmatization (e.g., *running → run*)  

- **Feature Extraction**:  
  Convert cleaned text into numerical vectors using **TF-IDF with n-grams (1,2)**.  

- **Model Training**:  
  Logistic Regression classifier trained on TF-IDF vectors.  

---

### Approach 2: Fine-Tuning a Transformer Model
This advanced approach uses **DistilBERT** to capture semantic and contextual meaning.

- **Model**: `distilbert-base-uncased` (smaller, faster BERT variant pre-trained on large corpora).  
- **Tokenization**: Uses DistilBERT tokenizer (no need for manual stop word removal).  
- **Fine-Tuning**: Model adapted on our review dataset via transfer learning.  
- **Training**: Performed on **Google Colab with GPU acceleration** for efficiency.  

## 4. Project Structure

```text
sentiment-analysis-project/
├── 📂 notebooks/
│   ├── 1_BoW_and_Logistic_Regression.ipynb    # Baseline model
│   └── 2_Transformer_Fine_Tuning.ipynb        # Advanced model
│
├── 📂 data/
│   └── K8 Reviews.csv                         # The raw dataset
│
├── 📂 saved_models/
│   ├── classifier.pickle                      # Saved Logistic Regression model
│   └── tfidfmodel.pickle                      # Saved TF-IDF vectorizer
│
└── 📜 README.md                               # Project documentation
```


## 5. How to View This Project
- **Step 1: Baseline** → Open `notebooks/1_BoW_and_Logistic_Regression.ipynb`  
  Runs on a standard local machine.  

- **Step 2: Advanced Model** → Open `notebooks/2_Transformer_Fine_Tuning.ipynb`  

📌 **Important Note**:  
This notebook requires a **GPU** for timely execution.  
- ~30 minutes on Google Colab (GPU)  
- ~30+ hours on standard CPU  

✅ Pre-saved training logs, performance plots, and the **final 89% accuracy** are available for direct viewing.  

---

## 6. Conclusion
This project successfully demonstrates the **evolution of NLP techniques** for sentiment analysis.  

- The **Bag-of-Words + Logistic Regression** approach provides a solid baseline.  
- The **fine-tuned Transformer (DistilBERT)** achieves **significantly higher accuracy** thanks to its **attention mechanism**, which enables better understanding of **context, nuance, and negation**.  

✨ This showcases the power of modern **deep learning models** in outperforming traditional methods for complex language tasks.
