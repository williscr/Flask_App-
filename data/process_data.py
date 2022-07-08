import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

<<<<<<< HEAD

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from CSV files , left join the files. 
    INPUT: 
    messages_filepath : Location of the messages csv 
    categories_filepath : Location of the categories csv 
  
    OUTPUT:
    df : dataframe 
    
    '''
    messages = pd.read_csv(messages_filepath)
    print(messages.columns)
    categories = pd.read_csv(categories_filepath)
    print(categories.columns)
    df = messages.merge(categories, how = 'left' , on = 'id')
    print('testtotal', df.columns)
=======
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left' , on = 'id')
>>>>>>> d67a6f7d145b5b71947df395dd5a249efcfea9e4
    return df 


def clean_data(df):
<<<<<<< HEAD
    '''
    Dataframe is cleaned. Correctly assign column names. Remove duplicates 
    INPUT: 
    df : Dataframe to be cleaned. 
    
    OUTPUT: 
    df: Cleaned dataframe 
    
    ''' 
=======
>>>>>>> d67a6f7d145b5b71947df395dd5a249efcfea9e4
    categories = df['categories'].str.split(';', expand=True)
    categories.columns=categories.iloc[0] 
    #Rename column names 
    categories.columns = pd.Index(map(lambda x : str(x)[:-2], categories.columns))
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

<<<<<<< HEAD
    df.drop(['categories'],axis = 1, inplace = True)
    df = pd.concat([df,categories],axis = 1)
       
=======
    df = df.drop(['categories'],axis = 1, inplace = True)
    df = pd.concat([df,categories],axis = 1)
    df.drop_duplicates(subset='id', inplace=True)
>>>>>>> d67a6f7d145b5b71947df395dd5a249efcfea9e4
    return df


def save_data(df, database_filename):
<<<<<<< HEAD
    '''
    Method to save dataframe to an SQLLITE database 
    INPUT: 
    df: Dataframe 
    database_filename : SQLLITE database name
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = "MessageCategories"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

=======
   table = database_filename.split('/')[1].split('.')[0]
   try:
        engine = create_engine(f'sqlite:///{database_filename}')
        df.to_sql(table, engine, index=False, if_exists='replace')
        return f"Successfully added {len(df)} rows to {database_filename}"
   except Exception as e: 
       return print(e)
      
>>>>>>> d67a6f7d145b5b71947df395dd5a249efcfea9e4


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()