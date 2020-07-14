import sys

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)

import pandas as pd
from pandas_summary import DataFrameSummary

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def load_df(file):
    """
    Takes a csv file and loads it into a dataframe.
    """
    df = pd.read_csv(file, usecols = ['Name', 'Sex', 'Age', 'Survived'])


def datacompleteness(df):
    """
    Takes a dataframe df and returns summary statistics for each column including 
    missing values.
    """
    df_summary = DataFrameSummary(df).summary()
    return df_summary



def plot_figures(df, col1, col2):
    """
    Takes a column in a dataframe (df) and plots seaborn figures.
    x = x axis column variable assigned to col1.
    y = y axis column variable assigned to col.
    """
    plotdata1 = df[col1]
    plotdata2 = df[col2]
    
    sns.catplot(x = plotdata1 , y = plotdata2, data = df, kind = 'swarm')


