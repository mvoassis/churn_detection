# Churn detection for a Telecom Operator.

Marcos Oliveira (mvoassis@gmail.com)


## Description

This project is part of Alura Challenges, where a set of analysis and Machine Learning models are applied to "Novexus," a fictional Telecom operator.

The objective is to help their team to decrease customer dropout level. In the initial meeting with the people responsible for the company's sales area, the importance of reducing the Customer Evasion Rate, also known as Churn Rate, was explained. After a meeting with the company's staff, they delivered a database with client and churn information, which should be analyzed, treated, and used to generate a classification model to identify potential dropout clients.

> Project developed using: Python, Jupyter Notebook (Google Colab), Pandas, Matplotlib, Seaborn, Scikit-learn, Numpy.

## Quick Access

* Exploratory Data Analysis - EDA (Notebook)
* .

## Files

* notebooks/Churn_prediction_EDA.ipynb - Exploratory data analysis and data cleaning/preparation notebook (Google Colab).
* data/data_clean.csv - Cleaned and treated dataset.

## Steps

### 1 - Data cleaning and Exploratory Data Analysis (EDA)

The data was providade as an API answer, structured as a multi-level JSON. The data was imported and normalized 

The following verifications and adjustments were performed regarding data cleaning:

  1. customerID is useless to a prediction method, should be droped.

  2. phone.MultipleLines, internet.OnlineSecurity, internet.OnlineBackup, internet.DeviceProtection, internet.TechSupport, internet.StreamingTV, internet.StreamingMovies: have "No XXX service" values, which  could be translated by "No".

  3. internet.InternetService, account.Contract, account.PaymentMethod: apply one-hot encoding.

  4. account.Charges.Total: convert to float

  5. Churn have empty values, that can be used on validation afterwards. Need to be removed from training.

  6. Churn, customer.gender, customer.Partner, customer.Dependents, phone.PhoneService, phone.MultipleLines, internet.OnlineSecurity, internet.OnlineBackup, internet.DeviceProtection, internet.TechSupport, internet.StreamingTV, internet.StreamingMovies,account.PaperlessBilling: convert to numeric binary using map().

  7. Normalize feature names.

Regarding the EDA, the following steps were performed:

1. Descriptive Analysis
2. Target variable distribution analysis.
3. Correlation Analysis
  * Target correlation;
  * Independent variables' correlation;
4. Boxplot distribution for numeric features;
5. Pairplot to visually highlight the distribution of the binary features.

> **Main EDA Insights:**

- There are 7032 non-null rows 27 features.
  - Most of them are binary features
  - Some were converted to binary using one-hot encoding.
  - Target feature ('Churn') is highly unbalanced (75/25)
  - Feature 'customer_gender' is highly balanced (50/50)

-----

- Correlation results did not show a strong coorelation between the target feature and any of the independent variables. 
- Multicollinearity where identified on the independent variables. However, these features were kept, since I intend to use tree-based models on the classifier. 

-----

- Mean values for the non-binary features "account_Charges_Monthly", "account_Charges_Total" and "customer_tenure" seems to influence the Churn status. (Boxplot)

-----

- The linear regression of the pairplots confirms the correlation results, visually highlighting the most relevant features regarding Churn values.  

### 2 - Classification Model Development

...

