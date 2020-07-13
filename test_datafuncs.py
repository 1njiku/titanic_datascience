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



def test_datacompleteness():
    """
    Takes a dataframe and checks for missing values.
    """
    df = pd.read_csv('exploration/data/titanic.csv', usecols = ['Name', 'Sex', 'Age', 'Survived'])
    df_summary = DataFrameSummary(df).summary()
    for col in df_summary.columns:
        assert df_summary.loc['missing', col] == 0, f'{col} has missing values'


def test_plotfigures():
    """
    Takes a column in a dataframe and plots seaborn figures.
    x = x axis column variable assigned to one column.
    y = y axis column variable assigned to second column.
    """
    df = pd.read_csv('exploration/data/titanic.csv', usecols = ['Name', 'Sex', 'Age', 'Survived'])
   

    
