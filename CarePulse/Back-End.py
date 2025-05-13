from flask import Flask, request, jsonify, g, render_template
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
DATABASE = 'medical_data.db'

# Ensure NLTK tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------- Database Helpers ----------
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db:
        db.close()

# ---------- Preprocessing ----------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# ---------- Response Logic ----------
def get_response(user_input, db_connection):
    cursor = db_connection.cursor()
    cursor.execute("SELECT symptom, advice FROM medical_info")
    medical_data = cursor.fetchall()

    symptom_texts = [row['symptom'] for row in medical_data]
    advice_lookup = {row['symptom']: row['advice'] for row in medical_data}

    raw_inputs = [part.strip() for part in user_input.lower().replace(',', ' and ').split(' and ') if part.strip()]

    responses = []
    for input_fragment in raw_inputs:
        all_texts = [*symptom_texts, input_fragment]
        processed_texts = [preprocess(t) for t in all_texts]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_texts)

        user_vec = tfidf_matrix[-1]
        symptom_vecs = tfidf_matrix[:-1]

        similarities = cosine_similarity(user_vec, symptom_vecs)[0]
        best_index = similarities.argmax()

        if similarities[best_index] >= 0.2:
            matched_symptom = symptom_texts[best_index]
            advice = advice_lookup[matched_symptom]
            if advice not in responses:
                responses.append(advice)

    if responses:
        return ' '.join(responses)
    else:
        return "I'm sorry, I don't have specific advice for that symptom right now. Please consult a healthcare professional."

# ---------- API Endpoint ----------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    if not data or not user_input:
        return jsonify({'error': 'No message provided'}), 400
    response = get_response(user_input, get_db())
    return jsonify({'response': response})

# ---------- Web Frontend ----------
@app.route('/')
def index():
    return render_template('index.html')

# ---------- Initialize Database ----------
if __name__ == '__main__':
    os.makedirs('instance', exist_ok=True)
    with app.app_context():
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symptom TEXT NOT NULL UNIQUE,
                advice TEXT NOT NULL
            )
        ''')

        symptom_advice_pairs = [
            ('headache', 'Rest and stay hydrated. If it persists, consult a doctor.'),
            ('migraine', 'Try to rest in a dark, quiet room. Over-the-counter pain relief may help.'),
            ('fever and cough', 'It could be a common cold or flu. Monitor your symptoms and consult a doctor if they worsen.'),
            ('sore throat', 'Gargle with warm salt water and drink warm fluids.'),
            ('stomach ache', 'Try eating light meals and stay hydrated. Avoid spicy food.'),
            ('nausea', 'Eat bland foods like crackers and stay hydrated.'),
            ('diarrhea', 'Drink lots of fluids and avoid dairy or greasy food.'),
            ('constipation', 'Eat fiber-rich foods and drink more water.'),
            ('chest pain', 'Seek immediate medical help as it could be serious.'),
            ('shortness of breath', 'This may be serious—seek medical attention immediately.'),
            ('runny nose', 'It could be allergies or a cold. Use tissues and stay hydrated.'),
            ('blocked nose', 'Try steam inhalation or decongestants.'),
            ('nasal congestion', 'Use a saline spray and stay hydrated.'),
            ('dizziness', 'Sit or lie down until it passes. Stay hydrated.'),
            ('fatigue', 'Get more rest and ensure you’re eating balanced meals.'),
            ('back pain', 'Apply a hot or cold compress and avoid heavy lifting.'),
            ('joint pain', 'Try gentle stretches and consider anti-inflammatory medication.'),
            ('rash', 'Avoid scratching and use anti-itch cream. See a doctor if it spreads.'),
            ('anxiety', 'Practice deep breathing or meditation. Talk to a mental health professional if needed.'),
            ('depression', 'Talk to someone you trust and consider professional support.'),
            ('sneezing and itchy eyes', 'Could be allergies. Antihistamines might help.')
        ]

        for symptom, advice in symptom_advice_pairs:
            cursor.execute("INSERT OR IGNORE INTO medical_info (symptom, advice) VALUES (?, ?)", (symptom, advice))

        conn.commit()
        close_db(None)

    app.run(debug=True)
