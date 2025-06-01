# EDA Package

A lightweight Python package for streamlined exploratory data analysis (EDA).  
It supports univariate, bivariate, and combined analysis with clean, informative visualizations using pandas, matplotlib, and seaborn.

## 📦 Features

- **`run_univariate_analysis(df_spark, features=None, top_k=5, plot=True)`**  
  Runs univariate analysis for both numeric and categorical features.

- **`run_bivariate_analysis(df_spark, target_col, cat_cols=None, num_cols=None,
                           num_bins=5, top_k=10, top_features=None, plot=True)`**  
  Analyzes relationships between a feature and a target variable.

- **`plot_univariate_bivariate(df_spark, target_col, features=None, top_k=10, num_bins=5)`**  
  Plots univariate and bivariate charts side by side for each feature (excluding the target_col).
  Converts binary integer features (with only two unique values) to strings for better categorical plots

## 🛠 Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/your-username/utils/eda_utils.git
cd eda_package
pip install -e .
