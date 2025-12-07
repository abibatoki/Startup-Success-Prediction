# 🚀 Predicting Startup Success with Crunchbase Data

> 🚀 **Best Model Accuracy (Random Forest with SMOTE): 87.9%**


## 📌 Project Overview

Driven by my interest in the fintech sector, I sought to take on a project that combines both business relevance and technical depth. This project focuses on predicting startup success using Crunchbase data. I approached it from three key angles:

- **Regression** to understand funding patterns  
- **Classification** to predict startup outcomes  
- **Clustering** to discover natural groupings among startups

The goal was to determine the factors that increase the likelihood of a startup's success — measured by IPOs, acquisitions, or a combination of both — while incorporating both machine learning techniques and business intelligence.

---

## 🧰 Code and Resources Used

- **Python Version:** 3.11  
- **IDE:** Jupyter Notebook  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `imbalanced-learn`  
- **Dataset:** An excerpt of a dataset released to the public by Crunchbase

---

## 🧹 Data Cleaning

- Filtered out companies younger than 3 years or older than 7 years to reduce noise (from 31,000+ to ~9,400 observations)
- Created a new `status` label:  
  - `Success` = Acquired, IPO, or both  
  - `Failure` = Closed or no exit
- Engineered `number_degrees` from MBA, PhD, MS, and Other
- Dropped columns with high missing values (e.g., `acquired_companies`, `products_number`)
- Handled missing values selectively
- Removed funding outliers (above 99th percentile)

---

## 📊 Exploratory Data Analysis (EDA)

EDA focused on understanding class balance and numerical relationships:
- **Status Value Counts:** Success vs Failure  
- **Log-Transformed Funding Distribution**  
- **Correlation Heatmap** of numeric variables

<img width="550" height="414" alt="image" src="https://github.com/user-attachments/assets/304c7ed0-a097-4467-a3d5-26edc2f935c1" />
  
<img width="535" height="419" alt="image" src="https://github.com/user-attachments/assets/7c1b2e08-22c6-4a5b-abb7-9860a86751bd" />
 
<img width="2692" height="2439" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/41924f47-46ea-42c6-87d4-77ad59227b7d" />

---

## 📈 Regression Modeling

### 🔹 Multiple Linear Regression (Statsmodels)

- Dependent variable: `average_funded`
- Significant predictors: `average_participants`, `number_degrees`, `ipo`
- `offices` and `is_acquired` were not statistically significant

<img width="483" height="458" alt="image" src="https://github.com/user-attachments/assets/2b74d209-7eab-48ee-b1d2-c02a3fe6b4fb" />

---

### 🔹 Linear Regression (Scikit-learn)

- Repeated regression using Scikit-learn’s `LinearRegression`
- Log-transformed `average_funded` for normality
- Coefficients confirmed the importance of `average_participants` and `ipo`

<img width="310" height="266" alt="image" src="https://github.com/user-attachments/assets/85a008e1-1fc3-4663-847d-83a214c56b91" />

---

## 🧪 Classification Modeling

### 🔸 Logistic Regression (Baseline – No SMOTE)

- Initially trained on imbalanced classes
- High overall accuracy but **poor recall for “Success”**
- Ineffective at identifying successful startups

<img width="345" height="203" alt="image" src="https://github.com/user-attachments/assets/10a5bf5f-d0b9-4ec4-9e1d-075e9e188edd" />

<img width="561" height="461" alt="image" src="https://github.com/user-attachments/assets/82d1c61d-2b54-4665-ac80-497eaffbe4e4" />
  
---

### 🔸 Logistic Regression (With SMOTE)

- Used SMOTE to balance the dataset
- Improved recall and precision for the “Success” class
- More reliable at flagging high-potential startups

<img width="353" height="202" alt="image" src="https://github.com/user-attachments/assets/339d0501-4e92-4a5e-bdcc-d7f53dd2c3cc" />

<img width="562" height="466" alt="image" src="https://github.com/user-attachments/assets/8c3a1ccb-a3a9-4e20-a67a-c6add1848609" />

---

## 🌲 Random Forest Classification

- Used SMOTE-balanced dataset
- Boosted accuracy from **70.7% to 87%**
- Strong performance across all evaluation metrics

<img width="352" height="202" alt="image" src="https://github.com/user-attachments/assets/f780d03a-75e4-4eb8-8329-5b7d26ba6c79" />

<img width="562" height="463" alt="image" src="https://github.com/user-attachments/assets/a3977da9-b9df-4bda-8773-682f6353fd89" />
    
<img width="2912" height="1665" alt="feature_importance" src="https://github.com/user-attachments/assets/0918348a-fde4-4f53-a79b-17711ed17488" />

---

## 🛠️ Hyperparameter Tuning

- Tuned `n_estimators`, `max_depth`, and `min_samples_split` using `GridSearchCV`
- Slight performance improvement
- Feature importance rankings remained stable

<img width="349" height="193" alt="image" src="https://github.com/user-attachments/assets/39a70358-7260-444c-91ad-8ee0ac554241" />

<img width="568" height="460" alt="image" src="https://github.com/user-attachments/assets/3e144dc0-1fc6-4026-8a0e-13282d28b6d1" />

---

## 📚 Clustering Analysis (KMeans)

For a final unsupervised learning step, I applied **KMeans Clustering** to group startups based on:

- `category_code` (encoded)
- `average_funded`
- `average_participants`

### Steps:
- Scaled all features using `StandardScaler`
- Used the **elbow method** to determine 3 optimal clusters
- Visualized results using 2D scatter plots

<img width="625" height="459" alt="image" src="https://github.com/user-attachments/assets/9680077d-76c7-4e44-a475-b5a6c13451b7" />

<img width="596" height="461" alt="image" src="https://github.com/user-attachments/assets/55191da8-29eb-417b-a485-56989580d2bc" />

<img width="603" height="463" alt="image" src="https://github.com/user-attachments/assets/7c36cd2b-66c0-44bc-96a4-2d5938e20813" />

### Key Insights:
- Startups in similar sectors tend to attract similar funding amounts
- Higher `average_participants` is associated with higher funding clusters
- These clusters help spot high-potential startups independent of IPO or acquisition status

---

## 💡 Final Thoughts

This end-to-end machine learning project gave me the chance to:

- Work with a real-world business dataset
- Build interpretable and high-performing models
- Apply data cleaning, feature engineering, and EDA effectively
- Handle class imbalance with SMOTE
- Improve model performance through hyperparameter tuning
- Add unsupervised clustering to surface hidden patterns

---




























