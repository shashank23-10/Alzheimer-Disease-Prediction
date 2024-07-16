from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


app = Flask(__name__)


# Load dataset
df = pd.read_csv('Alzheimer.csv')

# Preprocess the data
df.fillna(df.mode().iloc[0], inplace=True)
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Hand'] = df['Hand'].map({'R': 1, 'L': 0})

# Define features and target
X = df.drop(columns=['Subject ID', 'MRI ID', 'Group'])
y = df['Group']

clf = RandomForestClassifier()  # Instantiate the classifier
# Train the model
clf.fit(X, y)

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        mri = int(request.form['mri']) 
        visits = int(request.form['visits'])
        group = int(request.form['group'])
        educ = int(request.form['educ'])
        mmse = int(request.form['mmse'])
        etiv = int(request.form['etiv'])
        nwbv = int(request.form['nwbv'])
        # Include other form fields as needed

        # Make prediction
        input_data = [[age, sex, mri, visits, group, educ, mmse, etiv, nwbv]]  # Add other features here
        prediction = clf.predict(input_data)

        if prediction == 1:
            return render_template('index3.html', result = 'Positive chances of Alzheimers Disease')
        else:
            return render_template('index3.html', result = 'Negative')

# Save the model to a file
#joblib.dump(clf, 'my_model.pkl')

# Later, you can load the model using:
loaded_model = joblib.load('my_model.pkl')

if __name__ == '__main__':
    app.run(debug=True)
