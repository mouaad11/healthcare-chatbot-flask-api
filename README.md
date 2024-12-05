# Healthcare Symptom Diagnosis Chatbot

## Overview
A Flask-based medical chatbot that helps users diagnose potential health conditions based on their symptoms using machine learning.

## Features
- Interactive symptom-based diagnosis
- Decision Tree ML model for disease prediction
- Conversation flow management
- Detailed symptom and disease information

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/healthcare-chatbot.git
cd healthcare-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
Ensure the following directory structure:
```
project_root/
│
├── Data/
│   ├── Training.csv
│   └── Testing.csv
│
├── MasterData/
│   ├── symptom_Description.csv
│   ├── symptom_severity.csv
│   └── symptom_precaution.csv
```

### 5. Run the Application
```bash
python chat_bot.py
```

## API Endpoints

### Start Conversation
- **URL**: `/start-conversation`
- **Method**: POST
- **Response**: Returns a session ID and initial bot message

### Continue Conversation
- **URL**: `/continue-conversation`
- **Method**: POST
- **Required Params**: 
  - `session_id`
  - `message`

## Postman Testing

### 1. Start Conversation
- Create a POST request to `http://localhost:5000/start-conversation`
- Expected Response:
```json
{
    "session_id": "unique-uuid",
    "sender": "bot",
    "message": "Welcome to the HealthCare ChatBot! What is your name?",
    "is_first_message": true
}
```

### 2. Continue Conversation Flow
- Use the `session_id` from the first request
- Send messages sequentially:
  1. Your name
  2. Symptoms (e.g., "fever")
  3. Additional symptoms
  4. Number of days experiencing symptoms

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/Feature`)
3. Commit your changes (`git commit -m 'Add some Feature'`)
4. Push to the branch (`git push origin feature/Feature`)
5. Open a Pull Request

## License
Distributed under the MIT License.

## Contact
mouadaitahlal@gmail.com
