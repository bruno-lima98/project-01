import pandas as pd
import os
import pickle
from flask import Flask, request, jsonify
from preprocessing import preprocess_data

model_file = os.path.join(os.path.dirname(__file__), "model_C=0.1.bin")

with open(model_file, "rb") as f_in:
    model_wrapper = pickle.load(f_in)

app = Flask('startup_failure')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        startup = request.get_json()
        if not startup:
            return jsonify({'error': 'Empty or invalid JSON input'}), 400

        df_startup = pd.DataFrame([startup])
        df_proc, categorical, numerical = preprocess_data(df_startup)

        # Usar o scaler e dv do wrapper
        df_proc[numerical] = model_wrapper.scaler.transform(df_proc[numerical])
        X = model_wrapper.dv.transform(df_proc.to_dict(orient='records'))

        pred_proba = model_wrapper.model.predict_proba(X)[0, 1]
        pred_class = model_wrapper.model.predict(X)[0]

        result = {
            'failure_probability': float(pred_proba),
            'failure': bool(pred_class)
        }

        return jsonify(result)

    except Exception as e:
        print("❌ Erro durante a predição:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
