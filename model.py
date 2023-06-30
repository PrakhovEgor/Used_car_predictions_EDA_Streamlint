import pickle


def model_prediction(df):
    with open('data/model.pickle', 'rb') as f:
        model = pickle.load(f)
        return model.predict(df)
