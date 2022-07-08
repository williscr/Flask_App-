import sys
import pickle 
import pandas as pd 
import numpy as np
import re
import os

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])

def load_data(database_filepath, table_name = 'MessageCategories'):
    '''
    Function to load data from SQL Lite Database. 
    INPUT: 
    database_filename: Filepath for SQLite database file.
    OUTPUT: 
    X : Target Column (Independent Variables)
    Y : Dependent Variable
    ''' 
    
    engine = create_engine('sqlite:///' + database_filepath)
    text_df = pd.read_sql_table(table_name,engine)
    print(text_df.columns)
    
    X= text_df['message']
    Y = text_df.iloc[:,4:]
    
    return X , Y 




def tokenize(text):
    '''
    Function to clean and tokenize textual data
    INPUT 
    text: Textual input 
    OUTPUT : 
    clean_tokenized : Tokenized text data 
    
    
    ''' 
    #Remove stop words 
    #stop_words = stopwords.words("english")
    
    tokenize = word_tokenize(text); 
    # Clean the data 
    lemmatizer = WordNetLemmatizer() 
    
    clean_tokens = [] 
    
    for token in tokenize : 
        clean_token = lemmatizer.lemmatize(token).lower().strip() 
        clean_tokens.append(clean_token)
    
    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier. 
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
    
def build_model():
    ''' 
    Builds a classification model using the Sklearn library. 
    INPUT : NULL
    OUTPUT :
    cv : A skelearn model object 
    '''

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('txt_pipe', Pipeline([
                ('vectorise',CountVectorizer(tokenizer = tokenize)), 
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
    ])
    
    parameters = {
      'features__txt_pipe__tfidf__use_idf': (True, False)
       #'clf__estimator__n_estimators': [10, 50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv



def evaluate_model(model, X_test, Y_test):
    '''
    Calls classification report on the model returning the F1 score, recall and precision. 
    INPUT: 
    model : Model object 
    X_test : Test independent variable 
    Y_test : Test dependent variables 
    '''
    y_pred = model.predict(X_test)
    for idx, col in enumerate(Y_test):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))

  

def save_model(model, model_filepath):
    ''' 
    Export model as a pickle file
    INPUT: 
    model: The model you want to export
    model_filepath : Location of the model 
    ''' 
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()