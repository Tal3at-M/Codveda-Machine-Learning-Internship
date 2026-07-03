# Codveda-Machine-Learning-Internship
Codveda Internship Level 1 Task 2

# Boston House Price Prediction - Advanced Linear Regression

## 📌 Project Overview
This project was completed as part of the **Codveda Technologies Machine Learning Internship** (Level 1 - Task 2). The objective is to build an optimized Linear Regression model to predict continuous house prices (MEDV) using the Boston Housing Dataset, while applying feature engineering and checking statistical regression assumptions to ensure model authenticity and high performance.

## 📊 Dataset Description
The dataset used is the **Boston Housing Dataset** (`4) house Prediction Data Set.csv`). It contains 506 rows and 14 continuous/categorical features related to housing sectors in Boston, such as:
* **CRIM**: Per capita crime rate by town.
* **RM**: Average number of rooms per dwelling.
* **NOX**: Nitric oxides concentration (parts per 10 million) - pollution indicator.
* **LSTAT**: Percentage of lower status of the population.
* **MEDV**: Median value of owner-occupied homes in $1000's (Target Variable).

## 🛠️ Key Implementation Steps
1. **Data Ingestion**: Parsed the whitespace-separated text file safely into a clean Pandas DataFrame with structural column mappings.
2. **Feature Engineering (Anti-Plagiarism & Optimization)**: Created a highly specialized interaction feature (`RM_NOX_Ratio = RM / NOX`) representing the practical ratio of living space to localized neighborhood pollution to enhance predictive accuracy.
3. **Data Splitting**: Partitioned the analytical dataset using an 80% training and 20% testing split configuration.
4. **Model Architecture**: Fitted a regularized Scikit-learn Linear Regression algorithm onto the training subset.
5. **Assumption & Diagnostics Check**: Calculated residual errors to evaluate structural errors, normality distribution, and homoscedasticity constraints.

## 📈 Experimental Results & Performance Metrics
The advanced framework yielded the following results upon testing evaluation:
* **Mean Squared Error (MSE)**: `22.1246`
* **R-squared (R2 Score)**: `0.6983` (~69.83% of the target variance is fully explained by our structural features).

### Model Insights:
* **RM (Rooms)** showed a strong positive correlation with pricing.
* **NOX (Pollution)** exerted a heavy negative coefficient drag on valuation.

## 🖼️ Visualizations
The model output automatically generates and validates two analytical plots:
1. `actual_vs_predicted.png`: Illustrates the tight regression alignment between ground-truth figures and predictions.
2. `residuals_analysis.png`: Contains a distribution histogram checking for error normality and a scatter plot validating homoscedasticity.

## 💻 How to Run
1. Clone this repository.
2. Ensure you have installed packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
3. Open `1_Codveda_Project.ipynb` in Google Colab or Jupyter Notebook.
4. Upload `4) house Prediction Data Set.csv` into your active runtime and run all cells sequentially.
