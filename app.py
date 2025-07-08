from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("aquatic_disease_data.csv")
le_animal = LabelEncoder()
le_symptom = LabelEncoder()

df['Animal_enc'] = le_animal.fit_transform(df['Animal'])
df['Symptom_enc'] = le_symptom.fit_transform(df['Symptom'])

X = df[['Animal_enc', 'Symptom_enc']]
y = df['Disease']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    animals = sorted(df['Animal'].unique())
    symptoms = sorted(df['Symptom'].unique())
    return render_template("index.html", animals=animals, symptoms=symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    animal = request.form["animal"]
    symptom = request.form["symptom"]

    animal_enc = le_animal.transform([animal])[0]
    symptom_enc = le_symptom.transform([symptom])[0]

    prediction = model.predict([[animal_enc, symptom_enc]])[0]
    row = df[df['Disease'] == prediction].iloc[0]
    cure = row['Cure']
    medicine = row['Medicine']

    return render_template("result.html", animal=animal, symptom=symptom,
                           disease=prediction, cure=cure, medicine=medicine)

if __name__ == "__main__":
    app.run(debug=True)
