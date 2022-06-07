import sys
import pickle 
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


from sqlalchemy import create_engine

import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    table = database_filepath.split('/')[-1].split('.')[0]

    text_df = pd.read_sql_table(table,con=engine)
    
    X= text_df['message']
    Y = text_df.iloc[:,4:]
    
    return X , Y 



def tokenize(text):
    
    tokenize = word_tokenize(text); 
    # Clean the data 
    lemmatizer = WordNetLemmatizer() 
    
    clean_tokens = [] 
    for token in tokenize : 
        clean_token = lemmatizer.lemmatize(token).lower().strip() 
        clean_tokens.append(clean_token)
    
    return clean_tokens

def build_model():
    

    pipeline = Pipeline([
        ('vectorise',CountVectorizer(tokenizer = tokenize)), 
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
    ])


    parameters = {
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv



def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    for idx, col in enumerate(Y_test):
        print(col, classification_report(Y_test.iloc[:,idx], y_pred[:,idx]))

  

def save_model(model, model_filepath):
    
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