# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the placement dataset

2.Train Logistic Regression using Gradient Descent

3.Get student details from user and predict placement status

4.Generate the confusion matrix

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: T.Goshanrajan
RegisterNumber:  212225040098

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load dataset
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (1).csv")

# Convert categorical values to numeric
data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Select features and target
X = data[['gender', 'ssc_p', 'hsc_p', 'degree_p', 'mba_p']].values
y = data['status'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term
X = np.c_[np.ones(X.shape[0]), X]

# Initialize parameters
weights = np.zeros(X.shape[1])
learning_rate = 0.01
epochs = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent training
for _ in range(epochs):
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / y.size
    weights -= learning_rate * gradient

# ---------------- USER INPUT ----------------
print("\nEnter Student Details:")

gender = input("Gender (M/F): ")
gender = 1 if gender.upper() == 'M' else 0

ssc_p = float(input("SSC Percentage: "))
hsc_p = float(input("HSC Percentage: "))
degree_p = float(input("Degree Percentage: "))
mba_p = float(input("MBA Percentage: "))

# Normalize user input
user_input = np.array([gender, ssc_p, hsc_p, degree_p, mba_p])
user_input = (user_input - data[['gender','ssc_p','hsc_p','degree_p','mba_p']].mean().values) / \
             data[['gender','ssc_p','hsc_p','degree_p','mba_p']].std().values

# Add bias
user_input = np.insert(user_input, 0, 1)

# Prediction
result = sigmoid(np.dot(user_input, weights))

if result >= 0.5:
    print("\nPlacement Status: PLACED")
else:
    print("\nPlacement Status: NOT PLACED")

# ---------------- MODEL EVALUATION ----------------
y_pred = sigmoid(np.dot(X, weights)) >= 0.5

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("\nAccuracy:", accuracy)

# Precision
precision = precision_score(y, y_pred)
print("Precision:", precision)

# Sensitivity (Recall)
sensitivity = recall_score(y, y_pred)
print("Sensitivity (Recall):", sensitivity)

# Specificity
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP)
print("Specificity:", specificity)

```

## Output:
<img width="765" height="402" alt="image" src="https://github.com/user-attachments/assets/6244e231-130c-4604-9040-b436057c15a1" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

