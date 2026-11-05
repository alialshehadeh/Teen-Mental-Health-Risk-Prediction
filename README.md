# Teen Mental Health Risk Prediction Using Machine Learning

## Project Overview

This project develops a machine learning classification model to predict dataset-defined adolescent mental health risk categories: **Low, Medium, and High**.

The project explores relationships between social media usage, sleep patterns, academic performance, physical activity, stress, anxiety, and other lifestyle factors.

Six machine learning algorithms were compared, and **Random Forest** achieved the highest cross-validation Macro F1 score among the evaluated models.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool.

---

## Dataset Information

The original dataset was obtained from Kaggle.

- **Dataset:** Teen Social Media Usage and Mental Health
- **Creator:** sureshbeekhani
- **Original records used:** 2,500
- **Input features:** 11
- **Target variable:** `depression_risk`
- **Target classes:** Low, Medium, High

**Dataset Source:** https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded and inspected the dataset using Pandas.
2. Checked for missing values and duplicate records.
3. Cleaned the dataset.
4. Performed Exploratory Data Analysis (EDA).
5. Split the data into 70% training and 30% testing.
6. Applied OneHotEncoder to categorical features.
7. Used SMOTE to balance and augment the training dataset.

## Synthetic Data Generation Using SMOTE

The original dataset contained **2,500 records**. After splitting the dataset, 1,750 records were allocated for training and 750 records for testing.

To address class imbalance and investigate the effect of additional training samples, the Synthetic Minority Over-sampling Technique (SMOTE) was used.

SMOTE generates synthetic samples by interpolating between neighboring training observations within the same target class.

The training dataset was expanded from **1,750 to 10,000 samples**, consisting of 1,750 original training records and 8,250 synthetically generated records.

| Risk Category | Training Samples After SMOTE |
|---|---:|
| Low | 3,334 |
| Medium | 3,333 |
| High | 3,333 |
| **Total** | **10,000** |

SMOTE was applied only to the training data. The 750 testing records remained unchanged.

**Important:** The generated records are synthetic rather than independently collected real-world observations. Furthermore, applying standard SMOTE after one-hot encoding can create fractional values in categorical indicator columns. This is a methodological limitation that should be considered when interpreting the results.

## Exploratory Data Analysis

Data visualization was used to explore:

- Distribution of mental health risk categories.
- Age distribution.
- Social media usage patterns.
- Sleep duration.
- Stress and anxiety levels.
- Relationships between numerical features.
- Potential outliers using boxplots.

## Machine Learning Models Compared

Six classification algorithms were evaluated using five-fold stratified cross-validation.

| Model | CV Accuracy | CV Macro F1 |
|---|---:|---:|
| **Random Forest** | **77.89%** | **75.67%** |
| Gradient Boosting | 76.40% | 73.47% |
| Extra Trees | 75.43% | 71.37% |
| Logistic Regression | 74.57% | 70.81% |
| HistGradientBoosting | 74.51% | 70.41% |
| Decision Tree | 73.43% | 69.03% |

Random Forest was selected because it achieved the highest mean Macro F1 score among the models evaluated.

## Random Forest Model Training

The selected Random Forest model used the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 8 |
| min_samples_split | 10 |
| min_samples_leaf | 3 |
| max_features | sqrt |
| random_state | 42 |

## Model Performance

The effect of SMOTE was evaluated by comparing Random Forest trained on the original training data against Random Forest trained on the augmented dataset.

| Metric | Without SMOTE | With SMOTE |
|---|---:|---:|
| Training Records | 1,750 | 10,000 |
| Testing Accuracy | 75.87% | 78.27% |
| Testing Macro F1 | 72.50% | 76.06% |

**Observed improvements:**

- Accuracy increased by **2.40 percentage points**.
- Macro F1 increased by **3.56 percentage points**.

These improvements were observed on the project's existing test split. They do not establish that the improvement will generalize to independently collected data.

### Classification Performance After SMOTE

| Category | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| High | 0.70 | 0.95 | 0.81 |
| Low | 0.99 | 0.79 | 0.88 |
| Medium | 0.58 | 0.62 | 0.60 |

The Medium category remains the most challenging class to distinguish.

## Challenges and Solutions

| Challenge | Approach |
|---|---|
| Imbalanced target categories | SMOTE-based oversampling |
| Categorical input features | OneHotEncoder |
| Selecting an appropriate algorithm | Comparison of six classification models |
| Assessing model stability | Five-fold cross-validation |
| Optimizing model parameters | RandomizedSearchCV |
| Distinguishing Medium-risk records | Class-specific evaluation and error analysis |

## Limitations

- The project uses a relatively small original dataset.
- Synthetic samples are not equivalent to new real-world observations.
- Standard SMOTE applied after OneHotEncoder may generate unrealistic categorical combinations.
- The testing dataset was reused during experimentation, limiting its independence for final model selection.
- The model has not undergone external or clinical validation.
- Predictions should not be interpreted as medical diagnoses.

## Future Improvements

- Collect additional high-quality, representative real-world data.
- Compare SMOTE with categorical-aware oversampling methods such as SMOTENC.
- Improve classification performance for the Medium category.
- Evaluate generalization using an independent external test dataset.
- Investigate feature selection and model interpretability.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Google Colab
- Joblib

## References

1. Sureshbeekhani. *Teen Social Media Usage and Mental Health*. Kaggle.  
   https://www.kaggle.com/datasets/sureshbeekhani/teen-social-media-usage-and-mental-health

2. Scikit-learn Documentation.  
   https://scikit-learn.org/stable/

3. Imbalanced-learn Documentation.  
   https://imbalanced-learn.org/stable/

4. Pandas Documentation.  
   https://pandas.pydata.org/docs/

---

**Project Status:** Completed experimental machine learning pipeline with model comparison, synthetic data augmentation, cross-validation, and exploratory evaluation.
