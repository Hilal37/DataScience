import pandas as pd
import numpy as np

np.random.seed(0)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def dummy_ecoding_2_class(dfc, col):
  new_dfc = dfc
  first_class=pd.unique(new_dfc.iloc[:,col])[0]
  for i in range(len(dfc.iloc[:,col])):
    if dfc.iloc[i,col]==first_class:
      new_dfc.iloc[i,col]=1
    else:
      new_dfc.iloc[i,col]=0
  return new_dfc

df = pd.read_csv("train_features.csv")
X=dummy_ecoding_2_class(df,1)
X=dummy_ecoding_2_class(df,3)
#df_ohe.drop(columns = ['sig_id'], axis=1)
del X['sig_id']
pd.DataFrame(X)
y = pd.read_csv("train_targets_scored.csv")
del y['sig_id']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

model = RandomForestClassifier()

model.fit(X_train, Y_train)

import shap

shap_values = shap.treeExplainer(model).shap_values(X_train)
shap.summuray_plot(shap_values, X_train, plot_type = "bar")
