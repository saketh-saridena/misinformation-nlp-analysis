<div align="center">

<img src="assets/banner.svg" width="100%" alt="Misinformation NLP Analysis: should platforms answer for misinformation? 1,054 articles, mined.">

![Python](https://img.shields.io/badge/Python-0b1b33?style=flat-square&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-0b1b33?style=flat-square&logo=r&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-0b1b33?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Transformers-0b1b33?style=flat-square&logo=huggingface&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0b1b33?style=flat-square&logo=scikitlearn&logoColor=white)
![1,054 articles](https://img.shields.io/badge/articles-1%2C054-2f81f7?style=flat-square&labelColor=0b1b33)

</div>

**Should social media companies be held responsible for misinformation?** This text-mining project collects 1,054 news articles on the question from three sources. It labels each article's stance with a BERT model, then uses unsupervised learning to map what the coverage talks about and supervised learning to predict where an article stands.

## Highlights

- **Coverage leans against regulation.** 528 of the 1,054 articles (50%) argue against holding platforms responsible.
- **SVM reads sentiment best, at 68.72% accuracy.** Telling apart four stances is harder: 62.56% with an RBF kernel.
- **Five topics emerge.** LDA separates regulatory, technological, political and user-focused threads in the coverage.

<img src="assets/stance-labels.svg" width="100%" alt="Stance labels: 528 anti-regulation, 277 strongly pro-regulation, 189 weakly pro-regulation, 60 neutral">

## Results

<img src="assets/model-accuracy.svg" width="100%" alt="Accuracy: SVM RBF sentiment 68.72%, SVM RBF four stances 62.56%, SVM linear 59.24%, Naive Bayes 56.40%, best decision tree 51.66%">

With four uneven classes, chance level is about 50% (always guessing "anti-regulation"). So the stance models beat it by a modest margin, and the sentiment model does clearly better.

## Approach

| Step | Notebook | What happens |
| --- | --- | --- |
| 1. Collect and prepare | `01_collect_and_preprocess` | Pull articles from NewsAPI, MediaStack and Google News RSS; clean, tokenize, stem and lemmatize; build count and TF-IDF features; label stance with BERT |
| 2. Explore | `02a_kmeans_clustering`, `02b_hierarchical_clustering_r`, `02c_lda_topics`, `02d_association_rules.R` | K-Means with silhouette analysis, hierarchical clustering (cosine distance), LDA topic modeling, Apriori association rules |
| 3. Classify | `03_classification` | Naive Bayes, decision trees (entropy and Gini, varying depth), and SVM with linear, RBF and polynomial kernels |

## Repository

```
notebooks/   the analysis, in order (01 to 03)
data/        raw pulls from each source, the labeled set, and the processed text features
assets/      charts and banner used in this README
```

## Run it

The notebooks were written in Google Colab. Notebook 1 reads its API keys from the environment variables `NEWSAPI_KEY` and `MEDIASTACK_KEY`; both services offer free keys.

```
pip install -r requirements.txt
jupyter notebook notebooks/
```

The R parts need an R kernel (IRkernel) with `arules`, `arulesViz` and `tm`.
