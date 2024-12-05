import os
import re
import csv
import uuid
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full paths to CSV files
DATA_DIR = os.path.join(script_dir, 'Data')
MASTER_DIR = os.path.join(script_dir, 'MasterData')

TRAINING_PATH = os.path.join(DATA_DIR, 'Training.csv')
TESTING_PATH = os.path.join(DATA_DIR, 'Testing.csv')
SYMPTOM_DESCRIPTION_PATH = os.path.join(MASTER_DIR, 'symptom_Description.csv')
SYMPTOM_SEVERITY_PATH = os.path.join(MASTER_DIR, 'symptom_severity.csv')
SYMPTOM_PRECAUTION_PATH = os.path.join(MASTER_DIR, 'symptom_precaution.csv')

# Global variables for data and preprocessing
clf = None
le = None
cols = None
description_list = {}
severityDictionary = {}
precautionDictionary = {}

# Conversation sessions storage
conversation_sessions = {}

def safe_read_csv(file_path, key_column=None, value_column=None):
    """Safely read CSV files with optional key-value mapping."""
    result = {}
    try:
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            if key_column is not None and value_column is not None:
                for row in csv_reader:
                    if len(row) > max(key_column, value_column):
                        key = row[key_column]
                        value = row[value_column]
                        result[key] = value
            else:
                result = {row[0]: row[1] if len(row) > 1 else None for row in csv_reader}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return result

def load_medical_data():
    global clf, le, cols, description_list, severityDictionary, precautionDictionary
    
    try:
        # Read CSV files
        training = pd.read_csv(TRAINING_PATH)
        
        # Prepare columns
        cols = training.columns[:-1]
        x = training[cols]
        y = training['prognosis']

        # Encode labels
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # Train Decision Tree Classifier
        clf = DecisionTreeClassifier().fit(x_train, y_train)

        # Load additional dictionaries
        description_list = safe_read_csv(SYMPTOM_DESCRIPTION_PATH)
        
        severityDictionary = {k: int(v) for k, v in 
                               safe_read_csv(SYMPTOM_SEVERITY_PATH).items() 
                               if v.isdigit()}
        
        precautionDictionary = {row[0]: row[1:5] 
                                for row in csv.reader(open(SYMPTOM_PRECAUTION_PATH))}
    except Exception as e:
        print(f"Error loading medical data: {e}")
        raise

def check_pattern(dis_list, inp):
    """Find matching symptoms from input."""
    inp = inp.replace(' ', '_')
    pred_list = [item for item in dis_list if inp in item]
    return (1, pred_list) if pred_list else (0, [])

def sec_predict(symptoms_exp):
    """Secondary prediction using decision tree."""
    df = pd.read_csv(TRAINING_PATH)
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    
    for item in symptoms_exp:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])

def calc_condition(exp, days):
    """Calculate condition severity."""
    sum_severity = sum(severityDictionary.get(item, 0) for item in exp)
    severity_threshold = (sum_severity * days) / (len(exp) + 1)
    
    return "You should take consultation from a doctor." if severity_threshold > 13 else \
           "It might not be that bad, but you should take precautions."

def diagnose_symptoms(disease_input, symptoms_exp, num_days):
    """Diagnose symptoms and provide medical advice."""
    try:
        # Convert symptoms to match training data format
        symptoms_dict = {symptom: index for index, symptom in enumerate(cols)}
        input_vector = np.zeros(len(symptoms_dict))
        
        for item in symptoms_exp:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1

        # Use the global classifier for prediction
        prediction = clf.predict([input_vector])
        possible_diseases = list(le.inverse_transform(prediction))
        
        if not possible_diseases:
            possible_diseases = [disease_input or "Unknown"]
        
        primary_disease = possible_diseases[0]
        condition_advice = calc_condition(symptoms_exp, num_days)
        
        return {
            "primary_disease": primary_disease,
            "possible_diseases": possible_diseases,
            "condition_advice": condition_advice,
            "descriptions": {
                primary_disease: description_list.get(primary_disease, "No description available")
            },
            "precautions": precautionDictionary.get(primary_disease, [])
        }
    except Exception as e:
        print(f"Detailed diagnosis error: {e}")
        return {
            "primary_disease": disease_input or "Unknown",
            "possible_diseases": [disease_input or "Unknown"],
            "condition_advice": f"Diagnosis error: {e}. Please consult a healthcare professional.",
            "descriptions": {},
            "precautions": []
        }

@app.route('/start-conversation', methods=['POST'])
def start_conversation():
    """Initialize a new conversation session."""
    session_id = str(uuid.uuid4())
    
    initial_message = {
        "session_id": session_id,
        "sender": "bot",
        "message": "Welcome to the HealthCare ChatBot! What is your name?",
        "is_first_message": True
    }
    
    conversation_sessions[session_id] = {
        "name": None,
        "stage": "greeting",
        "symptoms": []
    }
    
    return jsonify(initial_message)

@app.route('/continue-conversation', methods=['POST'])
def continue_conversation():
    """Process conversation flow and user interactions."""
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message', '').strip()
    
    if session_id not in conversation_sessions:
        return jsonify({"error": "Invalid session"}), 400
    
    session = conversation_sessions[session_id]
    
    try:
        # Greeting Stage
        if session['stage'] == 'greeting':
            session['name'] = user_message
            session['stage'] = 'symptoms'
            return jsonify({
                "session_id": session_id,
                "sender": "bot",
                "message": f"Hello {user_message}! What symptoms are you experiencing?",
                "is_first_message": False
            })
        
        # Symptoms Stage
        if session['stage'] == 'symptoms':
            chk_dis = list(cols)
            conf, cnf_dis = check_pattern(chk_dis, user_message)
            
            if conf == 0:
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": "Enter a valid symptom.",
                    "is_first_message": False,
                    "suggestions": chk_dis
                })
            
            if len(cnf_dis) > 1:
                suggestion_list = "\n".join([f"{num}) {it}" for num, it in enumerate(cnf_dis)])
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": f"Searches related to input:\n{suggestion_list}\nSelect the one you meant (0 - {len(cnf_dis)-1}): ",
                    "is_first_message": False,
                    "symptom_options": cnf_dis
                })
            
            symptom = cnf_dis[0]
            session['symptoms'].append(symptom)
            session['stage'] = 'symptoms_confirmation'
            
            return jsonify({
                "session_id": session_id,
                "sender": "bot",
                "message": "Do you have any other symptoms? If not, type 'done'.",
                "is_first_message": False
            })
        
        # Symptoms Confirmation Stage
        if session['stage'] == 'symptoms_confirmation':
            if user_message.lower() == 'done':
                session['stage'] = 'days'
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": "For how many days have you been experiencing these symptoms?",
                    "is_first_message": False
                })
            
            chk_dis = list(cols)
            conf, cnf_dis = check_pattern(chk_dis, user_message)
            
            if conf == 0:
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": "Enter a valid symptom.",
                    "is_first_message": False,
                    "suggestions": chk_dis
                })
            
            if len(cnf_dis) > 1:
                suggestion_list = "\n".join([f"{num}) {it}" for num, it in enumerate(cnf_dis)])
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": f"Searches related to input:\n{suggestion_list}\nSelect the one you meant (0 - {len(cnf_dis)-1}): ",
                    "is_first_message": False,
                    "symptom_options": cnf_dis
                })
            
            symptom = cnf_dis[0]
            session['symptoms'].append(symptom)
            
            return jsonify({
                "session_id": session_id,
                "sender": "bot",
                "message": "Do you have any other symptoms? If not, type 'done'.",
                "is_first_message": False
            })
        
        # Days Stage
        if session['stage'] == 'days':
            try:
                num_days = int(user_message)
                result = diagnose_symptoms(
                    "Unknown", 
                    session['symptoms'], 
                    num_days
                )
                
                session['stage'] = 'complete'
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": f"Diagnosis:\n{result['condition_advice']}\n\n"
                              f"Possible Diseases: {', '.join(result['possible_diseases'])}\n"
                              f"Precautions: {', '.join(result['precautions'])}",
                    "is_first_message": False,
                    "diagnosis": result
                })
            except ValueError:
                return jsonify({
                    "session_id": session_id,
                    "sender": "bot",
                    "message": "Please enter a valid number of days.",
                    "is_first_message": False
                })
        
        return jsonify({
            "session_id": session_id,
            "sender": "bot",
            "message": "I'm not sure how to proceed. Let's start over.",
            "is_first_message": False
        })
    
    except Exception as e:
        return jsonify({
            "session_id": session_id,
            "sender": "bot",
            "message": f"An error occurred: {str(e)}",
            "is_first_message": False,
            "error": True
        }), 500

@app.before_request
def initialize_app():
    """Initialize medical data before each request."""
    if clf is None:
        load_medical_data()

if __name__ == '__main__':
    app.run(debug=True, port=5000)