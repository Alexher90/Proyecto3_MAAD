#!/usr/bin/python
import pandas as pd
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import os
def predict_proba(plot):
    plot = list(plot)
    plot = pd.Series(plot)    
    clf = joblib.load(os.path.dirname(__file__) + '/clasificador_P3.pkl') 
    words = pd.read_pickle(os.path.dirname(__file__) + '/words.pkl')
    Root = list(words[0])
    vect = CountVectorizer(vocabulary=Root , lowercase=False)
    X_dtm = vect.fit_transform(plot)
    p1 = clf.predict_proba(X_dtm)
    
    return p1
    
    
if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add Movie Plot')
        
    else:
        plot =      sys.argv[1]
        p1 = predict_proba(plot)
        print('Probability of Movie: ', p1)