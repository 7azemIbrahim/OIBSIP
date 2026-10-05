# Iris Species Classification

A machine learning project that explores the classic **Iris dataset** and trains multiple classifiers to predict flower species based on sepal and petal measurements.

## 📌 Overview

This project walks through a complete, beginner-friendly machine learning workflow:
- Loading and understanding the Iris dataset
- Exploratory Data Analysis (EDA) with visualizations
- Feature selection discussion
- Training and evaluating multiple classification models
- Comparing model performance and selecting the best one

## 📊 Dataset

The [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (Fisher, 1936), loaded via `sklearn.datasets.load_iris()`, contains:

- **150 samples**, 50 from each of 3 species
- **4 features** (all in cm): sepal length, sepal width, petal length, petal width
- **3 target classes**: `setosa`, `versicolor`, `virginica`
- No missing values; perfectly balanced classes

## 🗂️ Project Structure / Notebook Sections

1. **Load the Iris Dataset** — load data via `load_iris()` and inspect its structure (`data`, `target`, `target_names`, `feature_names`, `DESCR`)
2. **Understand the Features** — convert to a `pandas` DataFrame, check data types, missing values, summary statistics, and class balance
3. **Exploratory Data Analysis (EDA)** — deeper statistical and visual analysis of feature distributions and relationships
4. **Visualization** — histograms, class distribution, boxplots, violin plots, pairplot (scatter matrix), correlation heatmap, and outlier checks
5. **Feature Selection Discussion** — ranking features by how discriminative they are for classification
6. **Train/Test Split** — 80/20 stratified split using `train_test_split`
7. **Model Training** — trained three classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
8. **Model Evaluation** — accuracy score, confusion matrix, and classification report (precision, recall, F1) for each model
9. **Best-Performing Model** — comparison of all models and final model selection with justification

## 🔍 Key Findings

- **Petal length** and **petal width** are the most discriminative features (correlation with species ≈ 0.95–0.96); **sepal width** is the least discriminative.
- **Setosa** is perfectly linearly separable from the other two species across all models.
- **Versicolor** and **Virginica** show some overlap, which is where all three models made their few mistakes.
- All three models achieved **93.3% accuracy** (28/30 correct) on the test set, with slightly different error patterns:
  - Logistic Regression & Decision Tree: balanced, symmetric errors
  - KNN: biased toward predicting `versicolor` in ambiguous cases
- **Logistic Regression** was selected as the best-performing model — same top accuracy as the others, balanced errors, and simpler/more interpretable than KNN.

## 🛠️ Tech Stack

- Python
- `scikit-learn` — dataset, models, train/test split, metrics
- `pandas` — data manipulation
- `matplotlib` / `seaborn` — visualization

## ▶️ How to Run

1. Install dependencies:
```bash
   pip install scikit-learn pandas matplotlib seaborn
```
2. Open the notebook and run all cells in order (Load Data → EDA → Visualization → Train/Test Split → Model Training → Evaluation).

## 📈 Possible Next Steps

- Add cross-validation for a more robust model comparison
- Try additional classifiers (e.g., Random Forest, SVM)
- Hyperparameter tuning (e.g., `GridSearchCV` for KNN's `k` or Decision Tree depth)
