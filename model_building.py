import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load Dataset (Assuming titanic.csv is in your folder)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Preprocessing
# Select 5 features + target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# Handle Missing Values
X['Age'] = X['Age'].fillna(X['Age'].median())

# Encoding Categorical Variables
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Implement Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
print(classification_report(y_test, model.predict(X_test)))

# 6. Save Model
joblib.dump(model, 'model/titanic_survival_model.pkl')
print("Model saved successfully in /model/ folder!")