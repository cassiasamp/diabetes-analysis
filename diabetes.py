from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd


headers = ['times_pregnant', 'glucose', 'blood_pressure',
'skin_fold_thick', 'serum_insuling', 'mass_index',
'diabetes_pedigree', 'age']

with open(f'diabetes-model.pkl', 'rb') as f:
	model = pickle.load(f)


input_variables = pd.DataFrame([[1, 106, 70, 28, 135, 34.2, 0.142, 22]],
                                columns=headers, 
                                dtype=float,
                                index=['input'])

prediction = model.predict(input_variables)
print('Prediction ', prediction)

prediction_proba = model.predict_proba(input_variables)
print('Probabilities: ', prediction_proba)

app = Flask(__name__)
CORS(app)

@app.route("/katana-ml/api/v1.0/diabetes", methods=['POST'])
def predict():
    payload = request.json['data']
    values = [float(i) for i in payload.split(',')]
    
    input_variables = pd.DataFrame([values],
                                columns=headers, 
                                dtype=float,
                                index=['input'])

    # Get the model's prediction
    prediction_proba = model.predict_proba(input_variables)
    prediction = (prediction_proba[0])[1]
    
    ret = '{"prediction":' + str(float(prediction)) + '}'
    
    return ret

# running REST interface, port=5000 for direct test
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)

#tutorial: https://towardsdatascience.com/publishing-machine-learning-api-with-python-flask-98be46fb2440

