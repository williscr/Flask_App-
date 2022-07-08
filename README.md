# Disaster Response Pipeline Project

### Table of contents : 
1. Installation 
2. Project Motivation 
3. File Descriptions
4. Results 
5. Licencing Authors and Acknowledgements

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.  'python ru        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` used, include:

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


### Licencing , Authors , Acknowledgements 
Project done as part of the Udacity datascience nanodegree. 


#Screen Shot 


