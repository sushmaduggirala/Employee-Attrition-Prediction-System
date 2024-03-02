import pandas as pd
import numpy as np
import matplotlib.pyplot
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv ("watson_healthcare_modified.csv")
df.drop (['EmployeeID','Over18','StandardHours','EmployeeCount'],axis=1,inplace=True)


df['Attrition'] = df['Attrition'].apply (lambda x: 1 if x == "Yes" else 0)
df['OverTime'] = df['OverTime'].apply (lambda x: 1 if x == "Yes" else 0)

df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                            df['JobInvolvement'] +
                            df['JobSatisfaction'] +
                            df['RelationshipSatisfaction'] +
                            df['WorkLifeBalance']) / 5

df.drop (['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],
         axis=1,inplace=True)

df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply (lambda x: 1 if x >= 2.8 else 0)
df.drop ('Total_Satisfaction',axis=1,inplace=True)

df['Age_bool'] = df['Age'].apply (lambda x: 1 if x < 35 else 0)
df.drop ('Age',axis=1,inplace=True)

df['DailyRate_bool'] = df['DailyRate'].apply (lambda x: 1 if x < 800 else 0)
df.drop ('DailyRate',axis=1,inplace=True)

df['Department_bool'] = df['Department'].apply (lambda x: 1 if x == 'Maternity' else 0)
df.drop ('Department',axis=1,inplace=True)

df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply (lambda x: 1 if x > 5 else 0)
df.drop ('DistanceFromHome',axis=1,inplace=True)

df['JobRole_bool'] = df['JobRole'].apply (lambda x: 1 if x == 'Nurse' else 0)
df.drop ('JobRole',axis=1,inplace=True)

df['HourlyRate_bool'] = df['HourlyRate'].apply (lambda x: 1 if x < 65 else 0)
df.drop ('HourlyRate',axis=1,inplace=True)

df['MonthlyIncome_bool'] = df['MonthlyIncome'].apply (lambda x: 1 if x < 15000 else 0)
df.drop ('MonthlyIncome',axis=1,inplace=True)

df['NumCompaniesWorked_bool'] = df['NumCompaniesWorked'].apply (lambda x: 1 if x <3 else 0)
df.drop ('NumCompaniesWorked',axis=1,inplace=True)

df['TotalWorkingYears_bool'] = df['TotalWorkingYears'].apply (lambda x: 1 if x < 5 else 0)
df.drop ('TotalWorkingYears',axis=1,inplace=True)

df['YearsAtCompany_bool'] = df['YearsAtCompany'].apply (lambda x: 1 if x < 3 else 0)
df.drop ('YearsAtCompany',axis=1,inplace=True)

df['YearsInCurrentRole_bool'] = df['YearsInCurrentRole'].apply (lambda x: 1 if x < 3 else 0)
df.drop ('YearsInCurrentRole',axis=1,inplace=True)

df['YearsSinceLastPromotion_bool'] = df['YearsSinceLastPromotion'].apply (lambda x: 1 if x < 1 else 0)
df.drop ('YearsSinceLastPromotion',axis=1,inplace=True)

df['YearsWithCurrManager_bool'] = df['YearsWithCurrManager'].apply (lambda x: 1 if x < 1 else 0)
df.drop ('YearsWithCurrManager',axis=1,inplace=True)

df['Gender'] = df['Gender'].apply (lambda x: 1 if x == 'Female' else 0)

df.drop ('MonthlyRate',axis=1,inplace=True)
df.drop ('PercentSalaryHike',axis=1,inplace=True)

convert_category = ['BusinessTravel','Education','EducationField','MaritalStatus','Shift','OverTime',
                    'Gender','TrainingTimesLastYear']
for col in convert_category:
    df[col] = df[col].astype ('category')

X_categorical = df.select_dtypes (include=['category'])
X_numerical = df.select_dtypes (include=['int64'])
X_numerical.drop ('Attrition',axis=1,inplace=True)

y = df['Attrition']

onehotencoder = OneHotEncoder ()

X_categorical = onehotencoder.fit_transform (X_categorical).toarray ()
X_categorical = pd.DataFrame (X_categorical)
X_categorical


X_all = pd.concat ([X_categorical,X_numerical],axis=1)
X_all.head ()

X_train,X_test,y_train,y_test = train_test_split (X_all,y,test_size=0.20)

adaboost_classifier = AdaBoostClassifier()
adaboost_classifier.fit (X_train,y_train)

pickle.dump(adaboost_classifier, open('adamodel.pkl','wb'))
model = pickle.load(open('adamodel.pkl','rb'))