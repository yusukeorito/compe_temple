import itertools
import warnings

#from kaggler.preprocessing import TargetEncoder
import numpy as np
import pandas as pd  
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def count_encoding(train: pd.DataFrame, col_definition):
    for f in col_definition:
        count_map = train[f].value_counts().to_dict()
        train[f'ce_{f}'] = train[f].map(count_map)
    return train.loc[:, train.columns.str.contains('ce_')]

def create_numeric_feature(train, col_definition):
    return train[col_definition]


if __name__ == "__main__":
    train = pd.read_csv('../input/atmacup10_dataset/train.csv')
    test = pd.read_csv('../input/atmacup10_dataset/test.csv')
    print(train.shape, test.shape)

    categorical_cols = [
        'acquisition_method',
        'title',
        'principal_maker',

    ]

    numeric_cols = [
        'dating_period',
        'dating_year_early',
        'dating_year_late'
    ]
    

    target_cols = 'likes'

    train_test = pd.concat([train, test], axis=0).reset_index(drop=True)

    train_test[target_cols] = np.log1p(train_test[target_cols])

    #base
    train_test[[
        'acquisition_method',
        'title',
        'principal_maker',
        'dating_period',
        'dating_year_early',
        'dating_year_late'
        ] + [target_cols]].to_feather('../input/feather/train_test.ftr')

    #count_encoding
    count_encoding(train_test, categorical_cols).to_feather('../input/feather/count_encoding.ftr')
    create_numeric_feature(train_test,numeric_cols).to_feather('..input/feather/numeric_features.ftr')