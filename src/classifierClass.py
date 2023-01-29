import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


class Classifier:
    """
    Functional classification class, has one public method predict_from_code to predict the language.
    """
    def __init__(self):
        with open('Models/rfc_20%/best_rfc.pickle', 'rb') as data:
            self.model = pickle.load(data)

        with open('Pickles/tfidf_tokenless.pickle', 'rb') as data:
            self.tfidf = pickle.load(data)

    def _create_features_from_code(self, code):
        df = pd.DataFrame(columns=['file_body'])
        df.loc[0] = code

        df.replace('', np.nan, inplace=True)
        df.dropna(subset=['file_body'], inplace=True)

        return self.tfidf.transform(df['file_body']).toarray()

    def predict_from_code(self, code):
        """
        Method to classify provided snipped of code.

        :param code: (str) Code to classify
        :return: Void, prints predicted lagnuage and it's conditional probability.
        """
        f = self._create_features_from_code(code)
        # Predict using the input model
        prediction = self.model.predict(f)[0]
        prediction_prob = self.model.predict_proba(f)[0]
        print("The predicted language is", prediction)
        print(f"The conditional probability is: {(round(prediction_prob.max(), 5) * 100)}%")
