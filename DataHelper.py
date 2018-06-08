import pandas as pd
from sklearn.cross_validation import  train_test_split

data = pd.read_csv('points.csv')
# separate data from labels
X = data['X']
y = data['Y']
# split 80 20
# set random_state in order to maintain shuffle order and get same results on different runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=7)


def get_train_data():
    return X_train, y_train


def get_test_data():
    return X_test, y_test