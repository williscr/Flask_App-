# Disaster Response Pipeline Project

The aim of this project is to utilise skills learnt in the data engineering model and build a model for an API that classifier disaster messages. 
An image of the output to the flask app, classifiying a message can be seen below : 

![image](https://user-images.githubusercontent.com/48014687/177994705-8c66267d-9a68-4e30-9a58-6fe6ee9607d3.png)

=======
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.  'python        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` 

Libraries used, include:

- Database libraries : sqlite3 , SQLalchmey , pickle 
- General libries : pandas, numpy , re 
- Natural Language processing : NLTK 
- Machine Learning : Sklearn 
- Flask app and Visualisation : Flask, Plotly


### Project motivation 


### File Description 

├── app     
│   ├── run.py                           # Flask file that runs app
│   └── templates   
│       ├── go.html                      # Classification result page of web app
│       └── master.html                  # Main page of web app    
├── data                   
│   ├── disaster_categories.csv          # Dataset to process 
│   ├── disaster_messages.csv            # Dataset to process
│   └── process_data.py                  # Data cleaning process script 
├── models
│   ├── train_classifier.py              # ML model process script
|   └── classifier.pkl                   # Trained ML model
└── README.md

### Process_data.py

A pipeline to clean and format the data. It transforms the data by : 
Loading the datasets
Merging the messages and categories datasets
Cleans the data - I.e. Removes duplicates. 
Stores it in a SQLite database

### train_classifier.py

A machine learning pipeline that builds and trains the model 

- Loads data that was stored in a SQLite database in the ETL pipeline. 
- Partitions the data into training and testing.
- Builds a machine learning pipeline, to classify the messages into response categories 
- Trains and tunes the model using grid search techniques to obtain the optimal hyperparameters 
- Scores the models accuracy, f1 and precision 
- Exports the model to a pkl file. 

### run.py 

Flask app to visualise the input data and classifcation of data 

### Licencing , Authors , Acknowledgements 
Project done as part of the Udacity datascience nanodegree. 

