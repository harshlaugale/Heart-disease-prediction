from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import numpy as np

app = Flask(__name__)

# Load your dataset
data = pd.read_csv(r'C:\Users\Harshlaugale\OneDrive\Desktop\aiml mini project\Heart_Disease_Prediction.csv')  # Adjust this line to your dataset
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Calculate accuracies
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

# Save the model and accuracies
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(model, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    bp = float(request.form['bp'])
    cholesterol = float(request.form['cholesterol'])
    fbs = int(request.form['fbs'])
    ekg = int(request.form['ekg'])
    max_hr = int(request.form['max_hr'])
    exercise_angina = int(request.form['exercise_angina'])
    st_depression = float(request.form['st_depression'])
    slope_st = int(request.form['slope_st'])
    vessels_fluro = int(request.form['vessels_fluro'])
    thallium = int(request.form['thallium'])

    # Prepare input data for prediction
    input_data = np.array([[age, sex, cp, bp, cholesterol, fbs, ekg, max_hr, 
                            exercise_angina, st_depression, slope_st, 
                            vessels_fluro, thallium]])

    # Make prediction
    prediction = model.predict(input_data)
    result = "Heart Disease" if prediction[0] else "No Heart Disease"

    return render_template('result.html', result=prediction[0], train_accuracy=train_accuracy, test_accuracy=test_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
