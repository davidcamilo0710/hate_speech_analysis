# Hate Speech Analysis and Classification with NLP and Machine Learning

This repository contains a comprehensive project for the analysis and automatic detection of hate speech in text. The project is divided into two main parts:

1.  **Linguistic Analysis (hate-analysis):** A deep exploratory analysis to identify the features and patterns that distinguish hate speech from non-hate speech.
2.  **Machine Learning Classification (hate-classification):** The development and evaluation of supervised learning models to classify messages as "Hate" or "Non-Hate" based on the extracted features.

---

## Part 1: Linguistic Analysis and Feature Extraction

In this first phase, an exhaustive analysis was conducted on a corpus of **574,639 comments** to understand the properties of hate speech. **SpaCy** and the `es_core_news_md` model were used to process the text and extract linguistic features.

### Key Findings

The comparative analysis between hate comments (2.14% of the total) and non-hate comments (97.86%) revealed significant differences:

* **Message Length:** Hate comments are drastically shorter.
    * **Average words (Hate):** 15.60
    * **Average words (Non-Hate):** 107.14
* **Message Structure:** Hate messages are more direct and contain fewer sentences.
    * **Average sentences (Hate):** 1.55
    * **Average sentences (Non-Hate):** 3.99
* **Named Entity (NER) Usage:** Hate comments tend to be less specific and more generalized.
    * **% of comments with NER (Hate):** 36.52%
    * **% of comments with NER (Non-Hate):** 59.38%
    * Specifically, only **17.59%** of hate messages mention a person (`PERSON`), compared to **28.77%** in non-hate messages.
* **Lexicon Used:** There is a clear difference in vocabulary.
    * **Hate Lemmas:** Insults ("mierda", "puta", "asco"), pejorative terms ("gentuza", "miserable"), and highly charged political words ("gobierno", "fascista", "comunista") are predominant.
    * **Non-Hate Lemmas:** The focus is on informative and neutral topics ("año", "persona", "caso", "vacuna", "salud").
* **Morphology:** It was observed that hate speech contains a higher proportion of nouns and adjectives in the **masculine plural** (18.42%) compared to the feminine plural (9.26%), suggesting a focus on male collectives.

**Conclusion of Part 1:** The analysis demonstrated that there are quantifiable linguistic features (length, structure, lexicon, etc.) that act as strong indicators for differentiating hate speech, justifying their use in building a classification model.

---

## Part 2: Hate Speech Classification

Using the findings from the first part, a Machine Learning pipeline was built to classify messages. For this task, a balanced dataset of **10,000 comments** (50% Hate, 50% Non-Hate) with pre-extracted numerical features was used.

### Models and Evaluation

Three supervised classification algorithms were trained and compared:
1.  **Random Forest Classifier**
2.  **Support Vector Machine (SVM)**
3.  **XGBoost (Extreme Gradient Boosting)**

The models were evaluated using key metrics such as **F1-Score**, **Precision**, **Recall**, and **AUC-ROC**, as it is crucial in hate speech detection to balance the correct identification of toxic messages (Recall) and the avoidance of false accusations (Precision).

### Winning Model: XGBoost

The model with the best overall performance was **XGBoost**, trained on the original (non-standardized) features.

* **F1-Score:** **0.9811**
* **Accuracy:** 98.10%
* **Precision:** 97.72%
* **Recall:** 98.50%
* **AUC-ROC:** 0.9971

This means the model is capable of **detecting 98.5% of all hate speech messages**, with **97.7% of its alerts being correct**. In practice, for every 1,000 hate speech messages, the model would only fail to identify 15.

## Results Visualization

This is an excellent place to include the graphs from the classification notebook, as they visually summarize the models' performance.

#### Performance Comparison (F1-Score)

This graph shows that **XGBoost** and **Random Forest** achieve the best performance, and that data scaling (Standardized) negatively affects SVM.

<p align="center">
  <i>(Insert the F1-Score comparison image here)</i><br>
  <img src="URL_F1_SCORE_GRAPH" alt="F1-Score Comparison"/>
</p>

#### ROC Curves

The ROC curves demonstrate the excellent ability of all models to distinguish between the two classes, with AUC values very close to 1.

<p align="center">
  <i>(Insert the ROC Curves image here)</i><br>
  <img src="URL_ROC_CURVE_GRAPH" alt="Model ROC Curves"/>
</p>

---

## Code and Technologies

* **Linguistic Analysis Notebook:** **[hate-analysis.ipynb](URL_TO_ANALYSIS_NOTEBOOK)**
* **Classification Notebook:** **[hate-classification.ipynb](URL_TO_CLASSIFICATION_NOTEBOOK)**
* **Language:** Python
* **Core Libraries:**
    * **NLP Analysis:** SpaCy, Pandas
    * **Machine Learning:** Scikit-learn, XGBoost
    * **Visualization:** Matplotlib, Seaborn
