# Diabetes-Prediction-using-Machine-Learning

## Overview

This machine learning project focuses on binary classification to predict whether a person has diabetes or not using the PIMA Diabetes dataset. The Support Vector Machine (SVM) classifier with a linear kernel is employed for training and evaluation.

## Project Structure

### Importing the Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

## Data Collection and Analysis

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# displaying the first 5 rows of the dataset
print(diabetes_dataset.head())

# getting the number of rows and columns in the dataset
print(diabetes_dataset.shape)

# getting the statistical measures of the data
print(diabetes_dataset.describe())

# displaying the distribution of the target variable
print(diabetes_dataset['Outcome'].value_counts())
