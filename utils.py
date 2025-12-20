import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

cols_decimal = [
    'height',
    'weight'
    ]

cols_int = [
    'age',
    'fcvc',
    'ncp',
    'ch2o',
    'faf',
    'tue'
]

cols_int_unformatted = [
    'Age',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]

cols_categorical = [
    'gender', 
    'mtrans'
]

cols_ordinal = {
    'caec':['no', 'sometimes', 'frequently', 'always'],
    'calc':['no', 'sometimes', 'frequently', 'always'],
    'obesity':['insufficient_weight',
                'normal_weight',
                'overweight_level_i',
                'overweight_level_ii',
                'obesity_type_i',
                'obesity_type_ii',
                'obesity_type_iii']
}

cols_binary = [
    'family_history',
    'favc',
    'smoke',
    'scc'
]

cols_binary_unformatted = [
    ''
    'family_history',
    'FAVC',
    'SMOKE',
    'SCC'
]

dict_yn = {
    'yes': 1,
    'no': 0
}

class FormatColumnNames(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        df.columns = [col.lower().strip() for col in df.columns]
        return df

class FormatStrings(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.lower()
                df[col] = df[col].str.strip()
        return df

class ToFloat(BaseEstimator, TransformerMixin):
    def __init__(self,
                 to_float : list = cols_decimal) -> None:
        self.to_float = to_float
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        for col in self.to_float:
            df[col] = df[col].astype(float)
        return df

class ToInt(BaseEstimator, TransformerMixin):
    def __init__(self,
                 to_float : list = cols_int) -> None:
        self.to_float = to_float
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        for col in self.to_float:
            df[col] = df[col].round().astype(int)
        return df

class OneHot(BaseEstimator, TransformerMixin):
    def __init__(self,
                 to_onehot : list = cols_categorical) -> None:
        self.to_onehot = to_onehot
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        # create encoder
        enc = OneHotEncoder()

        # fit encoder
        enc.fit(df[self.to_onehot])

        # get the names of the new columns
        features = enc.get_feature_names_out(self.to_onehot)

        # make a dataframe with the new columns
        temp_df = pd.DataFrame(
            enc.transform(df[self.to_onehot]).astype(int).toarray(),
            columns = features,
            index = df.index
            )

        # concatenating with a slice of the original dataframe
        other_features = [feat for feat in df.columns if feat not in self.to_onehot]

        df_concat = pd.concat(
            [df[other_features],
            temp_df],
            axis = 1
        )

        return df_concat

class ToOrdinal(BaseEstimator, TransformerMixin):
    def __init__(self,
                 ordinal_dict : dict = cols_ordinal) -> None:
        self.ordinal_dict = ordinal_dict
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        for col in self.ordinal_dict:
            encoder = OrdinalEncoder(
                categories = [self.ordinal_dict[col]]
            )

            df[col] = encoder.fit_transform(df[[col]]).astype(int)

        return df


class ToBinary(BaseEstimator, TransformerMixin):
    def __init__(self,
                 to_binary : list = cols_binary) -> None:
        self.to_binary = to_binary
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        for col in self.to_binary:
            df[col] = df[col].map(dict_yn).astype(int)
        return df

class MakeObesityTarget(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self, df, y = None):
        return self
    
    def transform(self, df):
        df['target'] = df['obesity'] > 3
        df['target'] = df['target'].astype(int)

        return df

def apply_pipeline_ML(df):
    pipeline = Pipeline(steps = [
        ('format columns', FormatColumnNames()),
        ('format strings', FormatStrings()),
        ('to ordinal', ToOrdinal()),
        ('to float', ToFloat()),
        ('to int', ToInt()),
        ('to binary', ToBinary()),
        ('onehot', OneHot()),
        ('make target', MakeObesityTarget())
    ]
    )

    copy = df.copy()

    df_pipeline = pipeline.fit_transform(copy)

    return df_pipeline

def apply_pipeline_EDA(df):
    pipeline = Pipeline(steps = [
        ('format columns', FormatColumnNames()),
        ('format strings', FormatStrings()),
        ('to float', ToFloat()),
        ('to int', ToInt())
    ]
    )

    copy = df.copy()

    df_pipeline = pipeline.fit_transform(copy)

    return df_pipeline

