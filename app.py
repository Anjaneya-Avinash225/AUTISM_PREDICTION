from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import random
import string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import mysql.connector

# MySQL Database Configuration
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "admin@123",
    "database": "autism_db"
}

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = "your_secret_key"

# Dummy user database
users = {"admin": "password123", "user": "test123"}

# Load model
try:
    with open("autism_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

def generate_unique_id():
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=8))
    return letters + numbers

@app.route("/")
def home():
    return redirect(url_for("login"))

# ✅ Login route with JSON support
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if username in users and users[username] == password:
            session["user"] = username
            session["user_id"] = generate_unique_id()
            print("✅ Login successful! Redirecting to form...")
            return jsonify({"success": True, "redirect_url": url_for("patient_selection")})

        print("❌ Login failed. Invalid credentials.")
        return jsonify({"success": False, "error": "Login failed. Please check credentials."}), 401

    return render_template("login.html")

@app.route("/patient-selection")
def patient_selection():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("patient_selection.html")

@app.route("/select-patient", methods=["POST"])
def select_patient():
    data = request.json
    patient_type = data.get("patient_type")
    user_id = data.get("user_id")

    if patient_type == "new":
        session["user_id"] = generate_unique_id()
        return jsonify({"redirect": "/autism-test"})
    
    elif patient_type == "existing":
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        return jsonify({"redirect": f"/patient-details?user_id={user_id}"})

    return jsonify({"error": "Invalid selection"}), 400

@app.route("/get-patient", methods=["GET"])
def get_patient():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM patients WHERE user_id = %s", (user_id,))
        patient = cursor.fetchone()
        cursor.close()
        conn.close()

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        img_radar = io.BytesIO()
        categories = ["Social Interaction", "Eye Contact", "Repetitive Behavior", "Sensory Response"]
        values = [
            sum([patient.get("A1_Score", 0), patient.get("A3_Score", 0), patient.get("A5_Score", 0)]),
            sum([patient.get("A2_Score", 0), patient.get("A4_Score", 0), patient.get("A6_Score", 0)]),
            sum([patient.get("A7_Score", 0), patient.get("A9_Score", 0)]),
            sum([patient.get("A8_Score", 0), patient.get("A10_Score", 0)])
        ]
        values.append(values[0])
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values, linewidth=2, linestyle="solid")
        ax.fill(angles, values, alpha=0.4)
        plt.title("Autism Test Score Radar Chart", size=14)
        plt.tight_layout()
        plt.savefig(img_radar, format="png")
        img_radar.seek(0)
        patient["chart_radar"] = base64.b64encode(img_radar.getvalue()).decode()
        plt.close()
        return jsonify(patient)

    except mysql.connector.Error as err:
        print("❌ Database Error:", str(err))
        return jsonify({"error": "Database error occurred"}), 500

@app.route("/patient-details")
def patient_details():
    return render_template("patient_details.html")

@app.route("/get_user_id")
def get_user_id():
    session["user_id"] = generate_unique_id()
    return jsonify({"user_id": session["user_id"]})

@app.route("/autism-test")
def autism_test():
    if "user_id" not in session:
        return redirect(url_for("patient_selection"))
    return render_template("autism_test.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("📥 Received Data:", data)

        expected_features = ["age", "A1_Score", "A2_Score", "A3_Score", "A4_Score",
                             "A5_Score", "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score"]
        input_data = [int(data[feature]) for feature in expected_features]
        input_array = np.array([input_data], dtype=np.float32)

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict(input_array)[0]
        diagnosis = "Autistic" if prediction == 1 else "Not Autistic"

        return jsonify({
            "name": data.get("name", "Unknown User"),
            "prediction": diagnosis
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.json
        print("📥 Classification Data:", data)

        if "user_id" not in session:
            session["user_id"] = generate_unique_id()
        user_id = session["user_id"]

        name = data.get("name", "Unknown User")
        age = int(data.get("age", 0))
        numeric_fields = [
            "A1_Score", "A2_Score", "A3_Score", "A4_Score",
            "A5_Score", "A6_Score", "A7_Score", "A8_Score",
            "A9_Score", "A10_Score", "Routine_Change_Struggle",
            "Social_Anxiety", "Eye_Contact_Avoidance"
        ]
        scores = {field: int(data.get(field, 0)) for field in numeric_fields}
        answers = {q: data.get(q, "N/A") for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]}

        autism_type_mapping = {
            "A": "Kanner’s Syndrome",
            "B": "Asperger’s Syndrome",
            "C": "PDD-NOS",
            "D": "Childhood Disintegrative Disorder"
        }
        autism_type_counts = {t: 0 for t in autism_type_mapping.values()}
        for answer in answers.values():
            if answer in autism_type_mapping:
                autism_type_counts[autism_type_mapping[answer]] += 1

        autism_type = max(autism_type_counts, key=autism_type_counts.get)
        severe_symptoms = sum(1 for v in answers.values() if v in ["A", "D"])
        autism_level = "Severe" if severe_symptoms >= 3 else "Moderate" if severe_symptoms == 2 else "Mild"

        img_radar = io.BytesIO()
        categories = ["Social Difficulty", "Repetitive Behavior", "Sensory Sensitivity", "Communication Issues"]
        values = [
            sum([scores["A1_Score"], scores["A3_Score"], scores["A5_Score"], scores["Social_Anxiety"]]),
            sum([scores["A7_Score"], scores["A9_Score"], scores["Routine_Change_Struggle"]]),
            sum([scores["A8_Score"], scores["Eye_Contact_Avoidance"]]),
            sum([scores["A2_Score"], scores["A4_Score"], scores["A6_Score"]])
        ]
        values.append(values[0])
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values, linewidth=1, linestyle="solid")
        ax.fill(angles, values, alpha=0.4)
        plt.title(f"Autism Trait Profile for {name}")
        plt.savefig(img_radar, format="png")
        img_radar.seek(0)
        chart_radar = base64.b64encode(img_radar.getvalue()).decode()
        plt.close()

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()
            query = """
                INSERT INTO patients (user_id, name, age, 
                    A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, 
                    A7_Score, A8_Score, A9_Score, A10_Score, 
                    Routine_Change_Struggle, Social_Anxiety, Eye_Contact_Avoidance,
                    Q1, Q2, Q3, Q4, Q5, autism_type, autism_level) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                user_id, name, age,
                scores["A1_Score"], scores["A2_Score"], scores["A3_Score"], scores["A4_Score"],
                scores["A5_Score"], scores["A6_Score"], scores["A7_Score"], scores["A8_Score"],
                scores["A9_Score"], scores["A10_Score"], scores["Routine_Change_Struggle"],
                scores["Social_Anxiety"], scores["Eye_Contact_Avoidance"],
                answers["Q1"], answers["Q2"], answers["Q3"], answers["Q4"], answers["Q5"],
                autism_type, autism_level
            )
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Data saved successfully!")

        except mysql.connector.Error as e:
            print(f"❌ Database Error: {e}")

        return jsonify({
            "name": name,
            "user_id": user_id,
            "classification_result": {
                "autism_type": autism_type,
                "autism_level": autism_level
            },
            "chart_radar": chart_radar
        })

    except Exception as e:
        print("❌ Classification Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
