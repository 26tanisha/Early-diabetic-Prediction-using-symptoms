 # ![images_diabetics](https://github.com/user-attachments/assets/4bd0faf9-b584-484b-98bf-c5e7e8dd21f1)



## Early Diabetic Prediction Using Symptoms

# Problem Statement

Diabetes is one of the most common chronic diseases worldwide, and late diagnosis often leads to severe health complications such as heart disease, kidney failure, and vision loss.
Traditional diagnosis methods may require extensive medical tests, which can delay early detection. Therefore, there is a need for an intelligent system that can predict diabetes at an early stage using easily observable symptoms.

# Objective

The main objective of this project is to build a machine learning–based predictive system that can identify whether a person is likely to have diabetes based on clinical symptoms and health-related attributes.

Specific goals:
-To analyze symptom-based medical data
-To perform data preprocessing and exploratory data analysis
-To build and evaluate machine learning models for early diabetes prediction
-To assist in early diagnosis and preventive healthcare decision-making

# Techniques Used

This project uses the following data science and machine learning techniques:
Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA)
Label Encoding for categorical variables
Feature Scaling
Machine Learning Classification Algorithms
Model Evaluation using accuracy and other performance metrics

# Technologies & Tools

Programming Language: Python
Libraries:
    NumPy
    Pandas
    Matplotlib
    Seaborn
    Scikit-learn

# Dataset Description

The dataset contains medical and symptom-based information such as:
Polyuria
Polydipsia
Sudden weight loss
Weakness
Obesity
Age
Gender
Diabetes class (Positive / Negative)
The target variable is Class, which indicates whether a person is diabetic or not.

# Model Evaluation

The models are evaluated using:
Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
Special emphasis is given to Recall, as correctly identifying diabetic patients is critical in healthcare applications.

# Result
The final model is a support vector machine (SVM) with the SMOTE-ENN technique. The final model returned a recall score of 95.75% for the testing data.
