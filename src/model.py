import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0,inplace=True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(),inplace=True)
dataset['interview_score(out of 10)'].fillna(dataset['interview_score(out of 10)'].mean(),inplace=True)

X=dataset.iloc[:,:3]
y = dataset.iloc[:,-1]
dataset=dataset.applymap(lambda x : pd.to_numeric(x,errors='ignore'))

regressor=LinearRegression()

regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))