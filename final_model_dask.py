"""
Final Model: for learning purpose (dask version)

Author: Jasmine Guan and Sheng Yang
"""

# load packages
import os
from dask.distributed import Client
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, plot_confusion_matrix
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split  # dask version
from dask_ml.xgboost import XGBClassifier
from dask_ml.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# IO paths 
feature_path = "preprocess/feature.parquet/part*"
label_path = "preprocess/label.parquet/part*"
output_path = 'output_dask'


def initialize_client():
    """ initialize client """
    client = Client()
    return client


def read_feature_and_label():
    """
    read in the prepcrocessed feature and label dataset, and return X and Y 
    """
    # data import: load parquet file from feature_prep.py & label_prep.py
    feature_ddf = dd.read_parquet(feature_path)
    label_ddf = dd.read_parquet(label_path)
    # merge upon the unique identifier LoAN_SEQUENCE_NUMBER
    joined_ddf = feature_ddf.merge(label_ddf, on="LOAN_SEQUENCE_NUMBER", how="inner")
    X = joined_ddf.drop(columns=['LOAN_SEQUENCE_NUMBER', 'HARP_INDICATOR', 'label']).to_dask_array(lengths=True)
    y = joined_ddf["label"].to_dask_array(lengths=True)  # resolves chunksize issue
    return X, y


def split_and_pca(X, y):
    """
    perform train test split and PCA to dimension 5 
    """
    # train_test_split
    print('Performing Train Test Split')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True)
    # PCA 
    pca = PCA(n_components=5).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, y_train, y_test
    

# --- different models -----

def lr_train_test(X_train, X_test, y_train, y_test):
    """
    train and test using logistic regression, and log results into file 
    """
    # split and pca 
    print('Start Logistic Regression')

    # logistic regression pipeline
    lor = LogisticRegression(C = 10) 

    # fitting training set on Pipeline
    lor.fit(X_train, y_train)
    # make prediction & evaluate metric
    train_score = lor.score(X_train, y_train)
    test_score = lor.score(X_test, y_test)
    pred = lor.predict(X_test)
    f1 = f1_score(y_test, pred)

    # log outputs 
    log_model_performance('Logistic Regression', 
                          train_score=train_score, 
                          test_score=test_score, 
                          f1=f1
                         )

    # save plots 
    log_confusion_matrix('Logistic Regression', lor, X_test, y_test)


def xgb_train_test(X_train, X_test, y_train, y_test):
    """ train and test using xgboost classifier """
    print('Starting XGBoost')
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    
    # make prediction & evaluate metric
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    pred = clf.predict(X_test)
    f1 = f1_score(y_test, pred)

    # log outputs
    log_model_performance('XGBoost',
                          train_score=train_score,
                          test_score=test_score,
                          f1=f1
                          )

    # save plots
    log_confusion_matrix('XGBoost', clf, X_test, y_test)



# ---- for recording metrics ------ 

def log_model_performance(mdl_name, **kwargs):
    """
    record model performance in the output path
    
    :param mdl_name: the name of the model 
    """
    with open(os.path.join(output_path, 'accuracy_report.txt'), 'a') as f:
        f.write(f'{mdl_name}\n')
        for metric, value in kwargs.items():
            f.write(f'{metric}: {value}\n')
        f.write('\n')  # append an empty line


def log_confusion_matrix(mdl_name, mdl, X_test, y_test):
    """ save confusion matrix plot """
    plot_confusion_matrix(mdl, X_test, y_test, normalize='true', cmap='Blues')
    plt.title(f'{mdl_name} Test Set Confusion Matrix')
    plt.savefig(os.path.join(output_path, f'{mdl_name}_test_confusion_matrix.png'))


def main():
    """
    wrap all processes 
    """
    # initialize client
    initialize_client()
    # load features
    X, y = read_feature_and_label()

    # clear file
    with open(os.path.join(output_path, 'accuracy_report.txt'), 'w') as f:
        f.write('')

    # split and pca
    X_train, X_test, y_train, y_test = split_and_pca(X, y)

    # train and test on different models and record performances
    lr_train_test(X_train, X_test, y_train, y_test)
    xgb_train_test(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
