# Employee Attrition Prediction System

## Abstract
Many organizations invest significant resources and time in employee training. However, the departure of well-trained individuals can result in substantial financial losses and resource waste for the organization. The Employee Attrition Prediction System aims to assess an employee's likelihood of leaving the organization, enabling HR professionals to implement effective retention measures. The system employs a well-trained AdaBoost classifier, outperforming other classifiers with an accuracy of 93.15%. This project provides valuable insights into addressing employee attrition.

## Introduction
Attrition, the reduction in employees due to resignations or retirements, is a universal challenge for businesses. It incurs substantial costs, including business disruption, recruitment expenses, and training investments. The Employee Attrition Prediction System leverages historical data to proactively identify employees likely to leave, allowing organizations to implement retention measures and mitigate losses. This paper proposes a robust machine learning model trained on a "Employee Attrition for Healthcare" dataset sourced from Kaggle. AdaBoost emerged as the most effective model, with a training accuracy of 95.5% and a test accuracy of 93.15%.

## Project Overview

### Features
- Attrition prediction based on historical data.
- Comparative analysis with Decision Tree, Random Forest, XGBoost, and AdaBoost.
- Identification of key influencing attributes on attrition.

### Model Performance
- AdaBoost: Training Accuracy - 95.5%, Test Accuracy - 93.15%.
- Decision Tree: Test Accuracy - 87%.
- Random Forest: Test Accuracy - 91.15%.
- XGBoost: Test Accuracy - 91.69%.

### Key Influencing Attributes
- OverTime
- TotalWorkingYears
- JobLevel
- MonthlyIncome
- Age

## Getting Started
1. Clone the repository.
2. Install the required dependencies.
3. Explore the source code, datasets, and documentation.

## Usage
1. Train the model: Run train_model.py.
2. Evaluate model performance: Examine results in results/.
3. Predict attrition: Use the trained model for predictions.

## DataSet
From Kaggle: [Employee Attrition for Healthcare](https://www.kaggle.com/datasets/jpmiller/employee-attrition-for-healthcare)

## Authors
[Sushma Duggirala]

