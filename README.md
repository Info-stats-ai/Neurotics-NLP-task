# Neurotics -NLP-task and Fine tuning using Optuna

## Overview

This project explores various machine learning techniques to classify user text into neuroticism categories (Neurotic/Not Neurotic). The analysis involves text preprocessing, feature engineering using techniques like Noun Chunks, Empath, LIWC categories, TF-IDF, and Word2Vec embeddings, and training different classification models including Logistic Regression, Linear SVC, Non-Linear SVC, Multinomial Naive Bayes, Complement Naive Bayes, RandomForestClassifier, XGBoost, FastText, and BERT.

## Data

The project uses two datasets:
- `user_df`: Contains aggregated user data with a 'STATUS' column representing combined text posts and a 'cNEU' column as the target variable (0 for Not Neurotic, 1 for Neurotic).
- `df_test`: A separate test dataset for evaluation, containing 'TEXT' and 'cNEU' columns.

## Analysis and Modeling

The notebook covers the following steps:

1.  **Data Loading and Exploration**: Initial inspection of the datasets.
2.  **Feature Engineering**:
    *   Noun Chunk extraction using spaCy.
    *   Empath lexicon feature extraction.
    *   LIWC (Linguistic Inquiry and Word Count) category feature extraction.
    *   TF-IDF vectorization.
    *   Word2Vec embeddings (using pre-trained `glove-wiki-gigaword-100`).
    *   Combining text features with other numeric features (Big-5 scores, network metrics).
3.  **Model Training and Evaluation**:
    *   **Baseline Logistic Regression**: Using TF-IDF features.
    *   **Noun Chunks + Logistic Regression**: Using noun chunks as features.
    *   **Lemmatization + Logistic Regression**: Using lemmatized text with Logistic Regression.
    *   **Empath + Logistic Regression**: Combining Empath features with TF-IDF.
    *   **LIWC + Logistic Regression**: Combining LIWC features with TF-IDF.
    *   **Chi-Square Feature Selection**: Applying feature selection based on the chi-squared test.
    *   **PMI (Pointwise Mutual Information) Feature Selection**: Selecting features based on PMI scores.
    *   **Grid Search for Hyperparameter Tuning**: Optimizing Logistic Regression parameters using GridSearchCV.
    *   **SMOTE + Logistic Regression**: Handling class imbalance with SMOTE.
    *   **Word2Vec + Logistic Regression**: Using Word2Vec embeddings as features.
    *   **Linear SVC**: Training and tuning a Linear Support Vector Classifier.
    *   **Non-Linear SVC**: Training and tuning a non-linear Support Vector Classifier.
    *   **Naive Bayes Classifiers**: Training Multinomial and Complement Naive Bayes models.
    *   **RandomForestClassifier**: Training a Random Forest model.
    *   **XGBoost**: Training an XGBoost model.
    *   **FastText**: Training and evaluating a FastText model.
    *   **BERT (Bidirectional Encoder Representations from Transformers)**: Training a BERT model with class weights and exploring hyperparameter tuning with Optuna and using Focal Loss.
4. **Evaluation on a separate test dataset**: Evaluating the performance of selected models on `df_test`.


