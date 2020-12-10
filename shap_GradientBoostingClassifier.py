import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


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

df = pd.read_csv("train_features.csv",error_bad_lines=False)
X=dummy_ecoding_2_class(df,1)
X=dummy_ecoding_2_class(df,3)
#df_ohe.drop(columns = ['sig_id'], axis=1)
#del X['sig_id']
pd.DataFrame(X)
y = pd.read_csv("train_targets_scored.csv",error_bad_lines=False)
#del y['sig_id']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)

# fit model no training data
gboost = MultiOutputClassifier(GradientBoostingClassifier(random_state=0), n_jobs=1)
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
parameters = {'estimator__learning_rate': [0.005, 0.01, 0.1], 'estimator__n_estimators': [50, 100, 200]}
clf = GridSearchCV(gboost, parameters, cv=cv)

clf.fit(X_train, y_train)

import shap

shap_values = shap.TreeExplainer(clf).shap_values(X_train)
shap.summuray_plot(shap_values, X_train, plot_type = "bar")
