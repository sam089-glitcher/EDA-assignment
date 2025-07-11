# 📊 Netflix Customer Churn – Exploratory Data Analysis (EDA)

This project explores and analyzes customer data from Netflix to understand user behavior, subscription patterns, engagement levels, and churn tendencies.

## 🎯 Objective

- Analyze Netflix user data to identify behavioral patterns
- Investigate churn (customer cancellations) and its potential causes
- Visualize key metrics across demographics and plans
- Provide actionable business insights based on findings

---

## 📁 Dataset Overview

- **Total Records:** 5,000 customers  
- **Features:** 14 columns  
  - **Numerical:** `age`, `watch_hours`, `monthly_fee`, `last_login_days`, `number_of_profiles`, `avg_watch_time_per_day`, `churned`  
  - **Categorical:** `gender`, `subscription_type`, `region`, `device`, `payment_method`, `favorite_genre`  
- **Target Variable:** `churned` (0 = No churn, 1 = Churn)

---

## 📌 EDA Tasks Completed

### ✅ Data Overview
- Displayed first 5 rows
- Checked for null/missing values
- Reviewed unique values for all categorical columns

### ✅ Univariate Analysis
- Histograms for `age`, `watch_hours`, `monthly_fee`, `churned`
- Count plots for `subscription_type`, `gender`, `region`, `device`, `payment_method`, `favorite_genre`

### ✅ Bivariate Analysis
- Compared average `watch_hours` and `monthly_fee` across `subscription_type`, `region`, and `device`
- Average `watch_time_per_day` by `favorite_genre`
- Churn rates by `gender`, `region`, `subscription_type`, and `payment_method`

### ✅ Correlation Analysis
- Heatmap of numerical features
- Identified strong negative correlation between `watch_hours` and `churned`
- Positive correlation between `last_login_days` and `churned`

---

## 📊 Key Insights

1. **Lower watch hours = higher churn.**
2. **Premium users churn less than Basic users.**
3. **The West region has the highest average watch hours.**
4. **Inactivity (last login days) is a strong churn predictor.**
5. **TV users are more engaged and loyal than mobile users.**
6. **Higher monthly fee correlates with reduced churn.**
7. **Drama, Thriller, and Action are the most engaging genres.**

---

## 🧠 Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📂 Files

- `EDAassignment.py` – Main script containing all analysis steps
- `netflix_customer_churn.csv` – Dataset (not included here for privacy)
- Visual outputs and plots (generated within the script)

---
