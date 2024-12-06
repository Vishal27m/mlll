import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier

df=pd.read_csv("0")
df.dropna()
df.drop_duplicates()
imputer=SimpleImputer(strategy='mean')
df_impute=pd.DataFrame(imputer.fit_transform(df),columns=df.columns)
df['gender']=df['gender'].map({'male':0,'female':1})
X=df.drop(columns='outcome',axis=1)
y=df['outcome']

