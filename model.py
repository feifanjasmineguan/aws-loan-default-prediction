"""
model.py contains the necessary code to generate train and make predictions on
delinquency based on preprocessed features.

Author: Jasmine Guan and Sheng Yang 
"""

from sklearn.model_selection import train_test_split

feature_path = 'output/feature.parquet/'
