import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.metrics import f1_score


def load_data(rootdir='./'):
    x_train = np.loadtxt(rootdir + 'x_train.csv', delimiter=',').astype(int)
    y_train = np.loadtxt(rootdir + 'y_train.csv', delimiter=',').astype(int)
    x_test = np.loadtxt(rootdir + 'x_test.csv', delimiter=',').astype(int)
    y_test = np.loadtxt(rootdir + 'y_test.csv', delimiter=',').astype(int)
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    return x_train, y_train, x_test, y_test


def load_dictionary(rootdir='./'):
    county_dict = pd.read_csv(rootdir + 'county_facts_dictionary.csv')
    return county_dict


def print_dictonary(county_dict):
    for i in range(county_dict.shape[0]):
        print('Feature: {} - Description: {}'.format(i, county_dict['description'].iloc[i]))


def accuracy_score(preds, y):
    accuracy = (preds == y).sum() / len(y)
    return accuracy

def calc_f1(preds, y):
    same = (preds == y)
    not_same = (preds != y)
    tpp = (same == 1).sum()
    fpp = (not_same == 1).sum()
    tpn = (same == 0).sum()
    fnp = (not_same == 0).sum()
    percision = tpp / (tpp + fpp)
    recall = tpn / (tpn + fnp)
    return 2 * (percision * recall) / (percision + recall)


def f1(y, yhat):
    return f1_score(y, yhat)

###########################################################################
# you may add plotting or data processing functions (etc) in here if desired
###########################################################################
