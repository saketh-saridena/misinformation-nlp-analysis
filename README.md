# Social Media Misinformation: NLP Analysis & Classification

A comprehensive text mining and machine learning project investigating: **"Should social media companies be held responsible for misinformation?"** Analyzes 1,054 news articles using NLP, unsupervised learning, and supervised classification.

## Overview

Misinformation on social media has become a major concern, influencing elections, public health, and global events. This project collects, processes, and analyzes news articles to categorize stances on misinformation regulation using a full data science pipeline:

1. **Data Collection** from NewsAPI, MediaStack, and Google News RSS
2. **Text Preprocessing** with stemming, lemmatization, and tokenization
3. **Feature Engineering** using CountVectorizer and TF-IDF
4. **Unsupervised Learning** (K-Means, Hierarchical Clustering, LDA Topic Modeling, Association Rule Mining)
5. **Supervised Classification** (Naive Bayes, Decision Trees, SVM)

## Key Results

| Model | Accuracy |
|-------|----------|
| **SVM (RBF Kernel) - 4-class** | **62.56%** |
| **SVM (RBF) - Sentiment** | **68.72%** |
| SVM (Linear) | 59.24% |
| Naive Bayes | 56.40% |
| Decision Tree (best) | 51.66% |

**LDA Topic Modeling** identified 5 distinct topics spanning regulatory, technological, political, and user-focused perspectives on misinformation.

## Dataset

- **1,054 unique articles** collected from 3 news sources
- **Label Distribution:** Anti-regulation (528), Strongly pro-regulation (277), Weakly pro-regulation (189), Neutral (60)
- Labels generated using BERT sentiment classification

## Tech Stack

**Python:** pandas, scikit-learn, NLTK, Hugging Face Transformers (BERT), matplotlib, seaborn
**R:** arules, arulesViz, tm (Association Rule Mining)

## Project Structure

```
├── SakethSaridena_TextMining_Project_Part_1.ipynb    # Data collection & preprocessing
├── SakethSaridena_TextMining_Project_Part_2_*.ipynb   # Clustering (K-Means, Hierarchical, LDA)
├── SakethSaridena_TextMining_Project_Part_2_ARM.R     # Association Rule Mining
├── SakethSaridena_TextMining_Project_Part_3.ipynb     # Supervised classification (NB, DT, SVM)
└── Data/                                              # Raw & processed datasets (10 CSVs)
```

## Methods

### Unsupervised Learning
- **K-Means Clustering** (k=2-7, silhouette analysis)
- **Hierarchical Clustering** (cosine distance, average linkage)
- **LDA Topic Modeling** (5 topics, 100 iterations)
- **Association Rule Mining** (Apriori algorithm in R)

### Supervised Learning
- **Naive Bayes** (MultinomialNB)
- **Decision Trees** (3 variants with entropy/gini, varying depth)
- **SVM** (Linear, RBF, Polynomial kernels)

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas scikit-learn nltk transformers matplotlib seaborn`
3. Run notebooks in order (Part 1 -> Part 2 -> Part 3)
