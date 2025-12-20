import pandas as pd
import streamlit as st 

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import joblib

import psycopg2 as ps

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

import os


DATA_FOLDER = os.path.join('data','treated')
DATA_FILE = os.path.join(DATA_FOLDER, 'obesity-ml.csv')

MODEL_FOLDER = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_FOLDER, 'model.joblib')

df = pd.read_csv(DATA_FILE)

st.title("Análise e previsão de risco de Obesidade")