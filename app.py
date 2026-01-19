from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# 1. Load the trained model
# Using os.path to ensure it works on both local and cloud servers
MODEL_PATH = os.path.join('model', 'titanic_survival_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    """Renders the main GUI page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns prediction."""
    if request.method == 'POST':
        try:
            # 2. Extract data from the form (index.html names)
            pclass = int(request.form.get('Pclass'))
            sex = request.form.get('Sex')
            age = float(request.form.get('Age'))
            sibsp = int(request.form.get('SibSp'))
            parch = int(request.form.get('Parch'))

            # 3. Preprocess Categorical Data
            # Our model was trained with: male = 0, female = 1
            sex_val = 1 if sex.lower() == 'female' else 0

            # 4. Create a DataFrame for prediction
            # Column names MUST match the names used during model training
            input_features = pd.DataFrame(
                [[pclass, sex_val, age, sibsp, parch]], 
                columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
            )

            # 5. Make Prediction
            prediction = model.predict(input_features)[0]

            # 6. Format the output message
            if prediction == 1:
                result_text = "Result: The passenger likely SURVIVED."
            else:
                result_text = "Result: The passenger likely DID NOT SURVIVE."

            return render_template('index.html', prediction_text=result_text)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    # Use port 5000 for local testing
    app.run(debug=True)