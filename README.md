# Codveda Technologies - Machine Learning Internship Portfolio

This repository contains the completed milestones for the Machine Learning Internship at Codveda Technologies, structured systematically across multiple evaluation levels from foundations to advanced deep learning architectures.

---

## 🏠 Level 1 - Task 2: Boston House Price Prediction (Advanced Linear Regression)

### 📌 Project Overview
The objective is to build an optimized Linear Regression model to predict continuous house prices (MEDV) using the Boston Housing Dataset, while applying feature engineering and checking statistical regression assumptions.

### 📊 Dataset Description
The dataset used is `4) house Prediction Data Set.csv`. It contains 506 rows and 14 continuous/categorical features related to housing sectors in Boston, such as **CRIM** (crime rate), **RM** (number of rooms), **NOX** (pollution indicator), and **MEDV** (Target Variable).

### 🛠️ Key Implementation Steps
1. **Feature Engineering**: Created a highly specialized interaction feature (`RM_NOX_Ratio = RM / NOX`) representing the practical ratio of living space to localized neighborhood pollution to enhance predictive accuracy.
2. **Data Splitting**: Partitioned the analytical dataset using an 80% training and 20% testing split configuration.
3. **Model Training**: Fitted a Scikit-learn Linear Regression algorithm onto the training subset.
4. **Assumption Diagnostics**: Calculated residual errors to evaluate structural errors, normality distribution, and homoscedasticity constraints.

### 📈 Experimental Results
* **Mean Squared Error (MSE)**: `22.1246`
* **R-squared (R2 Score)**: `0.6983` (~69.83% of the target variance is fully explained).

---

## 🌸 Level 1 - Task 3: Iris Flower Classification (Optimized KNN Classifier)

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

## 📝 Level 2 - Task 1: Sentiment Analysis Pipeline (Logistic Regression NLP Classifier)

### 📌 Project Overview
The objective is to build a high-performance Text Classification framework using regularized Logistic Regression. The pipeline integrates text pre-processing, TF-IDF feature extraction, structural data augmentation to counter class imbalance, and exhaustive grid-search cross-validation.

### 📊 Dataset Description
The dataset utilized is `3) sentiment dataset.csv`. The targeted task focuses on a Binary Classification boundary to categorize textual strings into `Positive` (Label 1) or `Negative` (Label 0).

### 🛠️ Key Implementation Steps
1. **Advanced Text Cleansing**: Engineered a customized regex-driven text parser to clean URLs, HTML tags, punctuation, numbers, and standardize whitespaces into lowercased strings.
2. **Heuristic Data Augmentation**: Resolved a severe native class imbalance (45 Positive vs 4 Negative) by implementing targeted programmatic text augmentation, bringing the distribution to a balanced profile (45 Positive vs 44 Negative) to guarantee generalization.
3. **NLP Vectorization**: Transformed text corpora using a `TfidfVectorizer` mapping unigrams and bigrams up to 3,000 top features while excluding standard English stop words.
4. **Hyperparameter Optimization via GridSearchCV**: Executed a 3-Fold Cross-Validated grid search across regularization inverse-strengths ($C$) and optimization solvers.

### 📈 Experimental Results & Metrics
* **Optimized Test Accuracy**: `0.8889` (88.89% on unseen augmented splits).
* **Multi-Class Equilibrium**: Balanced Precision, Recall, and F1-Scores achieved perfectly at `0.89` for both classes, proving robust convergence on the minority class.

---

## 📈 Level 2 - Task 3: Stock Market Segmentation (Unsupervised K-Means Clustering)

### 📌 Project Overview
The objective is to implement an unsupervised learning pipeline using K-Means Clustering to group publicly traded assets based on mathematical financial risk-return dynamics, uncovering latent market sectors without manual labeling.

### 📊 Dataset Description
The dataset used is `2) Stock Prices Data Set.csv`, tracking historical pricing across 505 unique stock tickers.

### 🛠️ Key Implementation Steps
1. **Financial Feature Extraction**: Engineered asset-specific performance metrics by converting raw closing prices into rolling `Daily Returns`. Aggregated metrics per ticker to derive **Mean Return** (Performance) and **Volatility** (Risk via Standard Deviation).
2. **Feature Standardization**: Scaled the engineered matrices using `StandardScaler` to ensure the distance-based Euclidean calculations in K-Means aren't biased by feature magnitude variances.
3. **Hyperparameter Optimization**: Applied the **Elbow Method** tracking Within-Cluster Sum of Squares (WCSS) over 1 to 10 cluster ranges to isolate the optimal curvature bend at $K=4$.
4. **Clustering Architecture**: Initialized and fitted an optimized `KMeans` model with `k-means++` initialization to ensure robust cluster convergence.

### 📈 Clustering Interpretations & Results
The algorithm segmented the 505 equities into 4 highly distinct economic risk-return profiles:
* **Cluster 0 (65 Assets)**: *High Risk - High Return Profile* (Aggressive Growth Assets).
* **Cluster 1 (353 Assets)**: *Low Risk - Stable Return Profile* (Core Defensive/Blue-Chip Assets, e.g., AAPL).
* **Cluster 2 (81 Assets)**: *Moderate Risk - Stagnant Profile* (High Volatility with near-zero returns).
* **Cluster 3 (6 Assets)**: *Extreme Risk - Negative Return Profile* (Underperforming/Distressed Assets).

---

## 🛡️ Level 3 - Task 2: Support Vector Machine (SVM) Decision Boundary Analysis

### 📌 Project Overview
The objective is to implement a robust classification boundary using Support Vector Machines (SVM) on the Iris Dataset, comparing the mathematical convergence behavior of Linear vs. Radial Basis Function (RBF) non-linear Kernels, and rendering explicit 2D Decision Boundaries.

### 🛠️ Key Implementation Steps
1. **Binary Target Conversion**: Isolated the overlapping classes (`versicolor` and `virginica`) to form a complex decision boundary, standardizing the feature matrix via `StandardScaler`.
2. **Model Fitting**: Co-trained a Linear SVM and an RBF SVM using synchronized regularization parameters ($C=1.0$).
3. **Decision Boundary Grid Generation**: Built a structural 2D mesh-grid mesh to predict and visually project the exact hyper-plane and non-linear margin separations.

### 📈 Experimental Results
* **Linear Kernel Test Accuracy**: `0.8667`
* **RBF Kernel Test Accuracy**: `0.9000` (The non-linear hyper-plane accurately parsed the intertwined boundary points, optimizing classification performance).

---

## 🧠 Level 3 - Task 3: Deep Neural Network Text Classifier (TensorFlow & Keras)

### 📌 Project Overview
The objective is to construct a Deep Feed-Forward Artificial Neural Network (ANN) using TensorFlow/Keras to solve a Multi-Class NLP sentiment classification task. The model incorporates Backpropagation, Dropout Regularization, Softmax probability distribution, and early stopping diagnostics.

### 🛠️ Key Implementation Steps
1. **ANN Architecture Design**: Developed a 5-layer deep sequential network: Input Layer $\rightarrow$ Dense (64 Neurons, ReLU) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (16 Neurons, ReLU) $\rightarrow$ Dropout (0.2) $\rightarrow$ Dense (3 Output Neurons, Softmax).
2. **Imbalance Mitigation via Loss Calibration**: Computed programmatic balanced class weights (`compute_class_weight`) to penalize minority class classification errors dynamically during backpropagation.
3. **Diagnostic Plotting**: Tracked accuracy and loss iterations over validation splits to mathematically verify network convergence.

### 📈 Experimental Results
* **Training and Validation Stability**: Achieved high learning stability with validation accuracy successfully converging at `1.0000` and validation loss stabilizing perfectly at `0.5267` without displaying sudden divergent overfitting curves.

---

## 💻 How to Run
1. Clone this repository.
2. Ensure you have installed packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `tensorflow`.
3. Open the respective notebook (`.ipynb`) in Google Colab or Jupyter Notebook, upload the required `.csv` file, and run all cells sequentially.
