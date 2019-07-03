#DATASET: https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants

import pandas as pd
import numpy as np
import pickle
import random
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import OneHotEncoder


def where_highc(df, min=2, max=100):

    # This function is designed to identify the features of a Pandas dataframe
    #  with high cardinality

    num_cols = df.select_dtypes(include=[int, float]).columns
    cat_cols = list(set(df.columns) - set(num_cols))
    uniquec = [(x, len(df.loc[:, x].unique())) for x in cat_cols]
    highc = [(x[0], x[1]) for x in uniquec if max > x[1] > min]
    toohigh = [(x[0], x[1]) for x in uniquec if x[1] > max]

    return highc, toohigh


def applyW2V(df, columns):
    # Return specific columns in a dataframe as Word2Vector transforms
    assert type(columns) == list
    w2v = df.loc[:, columns].copy(deep=True)

    for col in columns:
        w2v[col] = w2v[col].astype('category')

    w2v = w2v.values.tolist()

    for i in w2v:
        random.shuffle(i)

    return Word2Vec(w2v)


# Read in and explore the data
filename = 'zomato.csv'
df = pd.read_csv(filename, dtype={'approx_cost(for two people)': str})
print("Length of the current dataset", len(df))
print("Counts of NaN values: ", df.isna().sum())
print("Counts of Null values: ", df.isnull().sum())

df['dish_liked'].replace(np.nan, "NA", inplace=True)
df.dropna(inplace=True)
print("length after NA drop: ", len(df))

df['cost'] = df['approx_cost(for two people)'].astype(str)
df['cost'] = df['cost'].apply(lambda x: float(x.replace(",", "")))
df['score'] = df['rate'].apply(lambda x: x.split("/")[0])
df['score'] = df['score'].apply(lambda x: 0.0 if x == 'NEW' or x == '-' else float(x))

# Get the high cardinality values to use Word2Vec on
highc, toohigh = where_highc(df)

# Apply Word2Vec to the dataframe
ccols = [x[0] for x in highc]
w2v = applyW2V(df, ccols)

pickle.dump(df, open("data.p", "wb"))


