# Churn detection for a Telecom Operator.

Marcos Oliveira (mvoassis@gmail.com)


## Description

This project is part of Alura Challenges, where a set of analysis and Machine Learning models are applied to "Novexus," a fictional Telecom operator.

The objective is to help their team to decrease customer dropout level. In the initial meeting with the people responsible for the company's sales area, the importance of reducing the Customer Evasion Rate, also known as Churn Rate, was explained. After a meeting with the company's staff, they delivered a database with client and churn information, which should be analyzed, treated, and used to generate a classification model to identify potential dropout clients.

> Project developed using: Python, Jupyter Notebook (Google Colab), Pandas, Matplotlib, Seaborn, Scikit-learn, Numpy.

## Quick Access

* [Exploratory Data Analysis - EDA (Notebook)](https://github.com/mvoassis/churn_detection/blob/main/notebooks/Churn_prediction_EDA.ipynb)
* [Classification model (Notebook)](https://github.com/mvoassis/churn_detection/blob/main/notebooks/Churn_detection_Classification_Model.ipynb)
* [Churn detector - WEB App (With operational classifier and EDA - Streamlit)](https://churndetection-mvoa.streamlit.app/)

## Files

* notebooks/Churn_prediction_EDA.ipynb - Exploratory data analysis and data cleaning/preparation notebook (Google Colab).
* data/data_clean.csv - Cleaned and treated dataset.

## Steps

### 1 - Data cleaning and Exploratory Data Analysis (EDA)

The data was providade as an API answer, structured as a multi-level JSON. The data was imported and normalized 

The following verifications and adjustments were performed regarding data cleaning:

  1. customerID is useless to a prediction method, should be droped.

  2. phone.MultipleLines, internet.OnlineSecurity, internet.OnlineBackup, internet.DeviceProtection, internet.TechSupport, internet.StreamingTV, internet.StreamingMovies: have "No XXX service" values, which  could be translated by "No".

  3. internet.InternetService, account.Contract, account.PaymentMethod: apply one-hot encoding? Since I intended to apply tree-based methods, I decided to use Target Encoder instead, which was implemented on the next step (Classification Notebook). 

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
- There seems to be a correlation between Churn and some informations of the categorical features, where some values are more present in Churn clients. They are:
  - Payment method = Electronic Check
  - Internet service = Fiber Optic
  - Contract type = Month-to-month 

-----

- The linear regression of the pairplots confirms the correlation results, visually highlighting the most relevant features regarding Churn values.  

### 2 - Classification Model Development

In this project, the main objective was to optimize the detection of Churn clients, so the company could take actions prevent it. Thus, the objective was to priorityze Recall, while trying to preserve a fair F1-Score. 

For classifying customers as churn or not churn, several machine learning classification algorithms were tested:

* Random Forest
* Gradient Boosting
* Logistic Regression
* XGBoost
* AdaBoost
* CatBoost
* Extra Trees
* Neural Network

The following steps were taken:

1. The data was split into train (80%) and test (20%) sets.

2. Target encoding was applied to the categorical variables in the train set.

3. The train set was then balanced using SMOTE oversampling.

4. Models were trained on the balanced train set and evaluated on the held-out test set.

5. Hyperparameter optimization was performed for the best performing model, AdaBoost, using BayesSearchCV.

6. The optimized AdaBoost model achieved 89% recall on the test set, with an F1-score of 57%.

> **Main Insights:**

Since the main objective of this project is to detect the as much Churn clients as possible, Recall was used to guide the hyperparameter optimization process.

Thus, the system achieved an 89% Recall rate on the test dataset.

Furthermore, although this result impaired the precision of the model, the overall F1-Score was 57%, just a bit worse than most baseline models. During the tests, no model achieved F1-Score higher than 63% even after hyperparameter optimization.

Some additional notes:

* The usage of Bayesian Search for hyperparameter optimization considerably reduced the amount of time spent os method's analysis.
* For the sake of organization, I decided to remove additional tests with other methods from this file, since they achieved inferior outcomes.
* Furthermore, I've investigated if a semi-supervised approach (clustering the data before applying to classifiers) would benefit the model. It did not. Even after using tSNE or PCA, the data visualization remained grouped, which shows that the data does could not be clusterized by k-means, BDSCAN or Mean-Shift methods.

### 3 - Churn Detector - Development of a WEB App

In this section, I will walk you through the implementation of the Churn Detector as a user-friendly web application using the **Streamlit framework**.

Application Overview
The Churn Detector web app serves as an intuitive tool for users to predict customer churn based on the insights gained from our data analysis and machine learning models. It allows users to input relevant customer information and receive predictions instantly. 

Streamlit proved to be an excellent choice for converting our analytical findings into a practical and accessible tool. Here's an overview of how I utilized Streamlit's capabilities:

1. User Interface (UI) Design: I designed a clean and user-friendly interface that enables users to input customer data effortlessly. Fields such as gender, partner status, dependents, and various service subscriptions can be easily filled out.

2. Prediction Engine: Behind the scenes, the machine learning model, trained on the cleaned and preprocessed data, powers the predictive functionality. Users can see the predicted churn status instantly after entering the customer's details.

3. Visualizations: To enhance user understanding, I integrated interactive visualizations within the app. Users can explore charts displaying relevant information, such as the distribution of churn predictions and categorical features distribution. 
