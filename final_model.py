"""
Final Model: use features and labels generated by feature_prep.py and 
label_prep.py to create prediction

Author: Jasmine Guan and Sheng Yang
"""

# load packages
import dask.dataframe as dd

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# data import: load parquet file from feature_prep.py & label_prep.py
feature_ddf = dd.read_parquet("output/feature.parquet/part*").compute()
label_ddf = dd.read_parquet("output/label.parquet/part*").compute()
joined_ddf = feature_ddf.merge(label_ddf, on = "LOAN_SEQUENCE_NUMBER", how = "inner").compute()
X = joined_ddf.drop("label", axis = 1)
y = joined_ddf[["label"]]

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# logistic regression pipeline
lor = LogisticRegression(C = 10)
pl = Pipeline(steps = [('logistic regression', lor)])

# fitting training set on Pipeline
pl.fit(X_train, y_train)

# make prediction & evaluate metric
pred = pl.predict(X_test)
f1 = metrics.f1_score(y_test, pred)
