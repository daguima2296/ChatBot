import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PretraitementMLP(BaseEstimator, TransformerMixin):
    """
    Transformer qui tokenise les commentaires d'un DataFrame, les transforme en séquences numériques,
    puis les remplit avec des zéros pour qu'ils aient tous la même longueur.
    """
    def __init__(self):
        pass
    
    def fit(self,X, y=None):
        """
        Entraîne le tokenizer sur la colonne 'commentaire' du DataFrame X.
        """
        self.tokenizer = Tokenizer(num_words=8000, oov_token='<00V>')
        # Recherche de la colonne 'commentaire' par son nom
        column_name = [col for col in X.columns if col.lower() == 'commentaireameliore']
        
        if column_name:
            self.tokenizer.fit_on_texts(X[column_name[0]])  # Accéder à la colonne par son nom
        else:
            print("La colonne 'commentaireAmeliore' n'a pas été trouvée dans le DataFrame.")
        return self
    
    def transform(self,X):
        """
        Tokenise les commentaires de la colonne 'commentaire' du DataFrame X,
        transforme chaque commentaire en séquence numérique
        """
        sequences = self.tokenizer.texts_to_sequences(X['commentaireAmeliore'])
        return sequences
      
class PretraitementMLPsuite(BaseEstimator, TransformerMixin):
    """
    Transformer qui tokenise les commentaires d'un DataFrame, les transforme en séquences numériques,
    puis les remplit avec des zéros pour qu'ils aient tous la même longueur.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        remplit chaque séquence avec des zéros pour qu'elle ait une longueur de 200.
        Retourne un tableau NumPy des données transformées.
        """
        padded = pad_sequences(X, maxlen=200, padding='post', truncating='post')
        return padded      
