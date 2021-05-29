# Final Model: use features and labels generated by feature_prep.py and label_prep.py to create prediction

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# data import: load parquet file from feature_prep.py & label_prep.py

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lor = LogisticRegression(C = 10)
pl = Pipeline(steps = [('logistic regression', lor)])

# fitting training set on Pipeline
# pl.fit(X_train, y_train)

# make prediction & evaluate metric
# preds = bpl.predict(X_test)
# f1 = metrics.f1_score(y_test, preds)