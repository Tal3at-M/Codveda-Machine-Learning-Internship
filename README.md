# Codveda Technologies - Machine Learning Internship (Level 1)

This repository contains the completed tasks for **Level 1 (Basic)** of the Machine Learning Internship at Codveda Technologies.

---

## 🏠 Task 2: Boston House Price Prediction (Advanced Linear Regression)

### 📌 Project Overview
The objective is to build an optimized Linear Regression model to predict continuous house prices (MEDV) using the Boston Housing Dataset, while applying feature engineering and checking statistical regression assumptions.

### 📊 Dataset Description
The dataset used is `4) house Prediction Data Set.csv`. It contains 506 rows and 14 features related to housing sectors in Boston, such as **CRIM** (crime rate), **RM** (number of rooms), **NOX** (pollution indicator), and **MEDV** (Target Variable).

### 🛠️ Key Implementation Steps
1. **Feature Engineering**: Created a highly specialized interaction feature (`RM_NOX_Ratio = RM / NOX`) representing the practical ratio of living space to localized neighborhood pollution to enhance predictive accuracy.
2. **Data Splitting**: Partitioned the analytical dataset using an 80% training and 20% testing split configuration.
3. **Model Training**: Fitted a Scikit-learn Linear Regression algorithm onto the training subset.
4. **Assumption Diagnostics**: Calculated residual errors to evaluate structural errors, normality distribution, and homoscedasticity constraints.

### 📈 Experimental Results
* **Mean Squared Error (MSE)**: `22.1246`
* **R-squared (R2 Score)**: `0.6983` (~69.83% of the target variance is fully explained).

---

## 🌸 Task 3: Iris Flower Classification (Optimized KNN Classifier)

### 📌 Project Overview
The objective is to build an optimized K-Nearest Neighbors (KNN) Classifier to accurately classify Iris flower species based on structural measurements, while implementing feature scaling, 5-Fold Cross-Validation, and comprehensive evaluation metrics.

### 📊 Dataset Description
The dataset used is `1) iris.csv`. It contains 150 instances and 5 attributes: **sepal_length**, **sepal_width**, **petal_length**, **petal_width**, and **species** (Target categorical class: `setosa`, `versicolor`, `virginica`).

### 🛠️ Key Implementation Steps
1. **Advanced Feature Engineering**: Created a custom interactive ratio feature (`Petal_Ratio = petal_length / petal_width`) to distinguish complex boundaries and ensure code authenticity.
2. **Feature Scaling**: Applied `StandardScaler` to normalize numerical variables to prevent distance-based scaling bias in the KNN algorithm.
3. **Hyperparameter Tuning via Cross-Validation**: Evaluated K values from 1 to 14 using a 5-Fold Cross-Validation pipeline (`cross_val_score`) to find the mathematically optimal $K=3$.
4. **Model Training**: Trained a final Scikit-learn `KNeighborsClassifier` configured at $K=3$.

### 📈 Experimental Results
* **Final KNN Model Accuracy**: `1.0000` (100% accuracy on the 30-sample testing split).

> 🔍 **Overfitting Validation Note**: While a perfect score of 1.0000 often signals Overfitting, it is a mathematically verified and expected result here due to high cluster separability in the feature space and confirmed stability via the 5-Fold Cross-Validation tuning loop.

---

## 💻 How to Run
1. Clone this repository.
2. Ensure you have installed packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
3. Open the respective notebook (`.ipynb`) in Google Colab or Jupyter Notebook, upload the required `.csv` file, and run all cells sequentially.
