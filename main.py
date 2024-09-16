from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from scipy.sparse import hstack
app = Flask(__name__)
model = joblib.load('model.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')
import traceback
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        df = pd.DataFrame([data])
        df = df[['amt', 'merchant', 'category', 'gender', 'state', 'zip','lat', 'long', 'job', 'is_fraud']]
        categorical_columns = ['merchant', 'category', 'gender', 'state', 'zip', 'job']
        numerical_columns = ['amt', 'lat', 'long']
        X_test = df[['amt', 'merchant', 'category', 'gender', 'state', 'zip','lat', 'long', 'job']]
        X_test_encoded = onehot_encoder.fit_transform(df[categorical_columns])
        X_test_scaled = scaler.fit_transform(df[numerical_columns])
        X_test_final = hstack((X_test_encoded, X_test_scaled))
        prediction = model.predict(X_test_final)
        if(prediction==0):
            return jsonify({'prediction': "Not a Fraud"})
        return jsonify({'prediction': "Fraud"})
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred during prediction.'}), 400
if __name__ == '__main__':
    app.run(debug=True)
