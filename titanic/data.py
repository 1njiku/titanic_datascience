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

"""
def load_df(file):
    
    Takes a csv file and loads it into a dataframe.
    
    df = pd.read_csv(file, usecols = ['Name', 'Sex', 'Age', 'Survived'])
"""

def datacompleteness(df):
    """
    Takes a dataframe df and returns summary statistics for each column including 
    missing values.
    """
    df_summary = DataFrameSummary(df).summary()
    return df_summary


"""
def plot_figures(df, col1, col2):
    
    Takes a column in a dataframe (df) and plots seaborn figures.
    x = x axis column variable assigned to col1.
    y = y axis column variable assigned to col.
    
    plotdata1 = df[col1]
    plotdata2 = df[col2]
    
    sns.catplot(x = plotdata1 , y = plotdata2, data = df, kind = 'swarm')
"""


def extract_title(df):
    """Extract the title from the passenger names.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing the column `Name`
    Returns
    -------
    pandas.DataFrame
        Data-frame with additional column with titles
    """

    logging.info("Extracting the titles from the name column")

    simplify_title = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    title = df['Name'].apply(
        lambda full_name: (
            simplify_title[
                # Example: Uruchurtu, Don. Manuel E --> Don
                full_name.split(',')[1].split('.')[0].strip()
            ]
        )
    )

    merged = df.merge(title.to_frame(name='Title'),
                      left_index=True, right_index=True)
    merged['Title'] = merged['Title'].astype('category')

    return merged

