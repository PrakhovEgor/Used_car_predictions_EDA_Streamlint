import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd

def model_prediction(df):
    with open('data/model.pickle', 'rb') as f:
        model = pickle.load(f)
        return model.predict(df)