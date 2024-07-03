# CREDIT-CARD-FRAUD-DETECTION

In this project, we aim to build a machine learning model to identify fraudulent credit card transactions. We'll follow these steps:

Load the Data: Load the credit card transaction dataset.
Explore the Data: Examine the dataset to understand its structure and contents.
Preprocess the Data: Clean the data, normalize it, and handle class imbalance.
Split the Data: Split the dataset into training and testing sets.
Model Building: Train a classification algorithm.
Model Evaluation: Evaluate the model's performance using precision, recall, and F1-score.
Step 1: Load the Data
We'll use a publicly available dataset from Kaggle. This dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

Step 2: Explore the Data
We'll examine the dataset to understand its structure and contents. This includes checking for missing values, understanding the distribution of features, and getting a sense of the class imbalance.

Step 3: Preprocess the Data
This step includes:

Normalizing the Data: Using StandardScaler to normalize the feature values.
Handling Class Imbalance: Using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.
Step 4: Split the Data
We'll split the data into training and testing sets to evaluate the performance of our model.

Step 5: Model Building
We'll train a RandomForestClassifier to classify the transactions. Other classifiers like Logistic Regression can also be used.

Step 6: Model Evaluation
We'll evaluate the model's performance using precision, recall, and F1-score, which are appropriate metrics for imbalanced datasets.

Here is the complete implementation:

python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Step 1: Load the Data
data_url = "https://raw.githubusercontent.com/sonu0516/Titanic-Survival-Prediction/master/creditcard.csv"
df = pd.read_csv(data_url)

# Step 2: Explore the Data
print(df.head())
print(df.describe())
print(df['Class'].value_counts())

# Check for missing values
print(df.isnull().sum())

# Step 3: Preprocess the Data
# Normalize the features
scaler = StandardScaler()
X = df.drop(columns=['Class'])
y = df['Class']
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Classification Report:\n{report}")
Explanation:
Load the Data: We load the dataset using pandas' read_csv function.
Explore the Data: We examine the first few rows, get descriptive statistics, and check the class distribution. We also check for any missing values.
Preprocess the Data:
Normalization: We normalize the feature values using StandardScaler.
Class Imbalance Handling: We use SMOTE to balance the classes by generating synthetic samples for the minority class.
Split the Data: We split the data into training and testing sets using an 80-20 split.
Model Building: We train a RandomForestClassifier on the training set.
Model Evaluation: We evaluate the model's performance using precision, recall, and F1-score. The classification report provides a detailed breakdown of these metrics for both classes.
