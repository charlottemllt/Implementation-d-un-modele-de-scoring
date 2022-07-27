from flask import Flask, render_template, jsonify, request, flash
import joblib
import pandas as pd

model = joblib.load('LGBM.joblib').best_estimator_

url_df = 'https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/API/df_API_lite.csv'
df_test = pd.read_csv(url_df)

liste_clients = list(df_test['SK_ID_CURR'].sort_values())

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/predict/<int:ID_client>', methods=["POST", "GET"])
def predict_client(ID_client):
    seuil = 0.91

    if ID_client in liste_clients:
        X = df_test[df_test['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = model.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)

        if predictions_proba[:, 1] >= seuil:
            predict = 1
        else:
            predict = 0
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify({ 'retour_prediction' : str(predict),
                     'predict_proba_0': str(predict_proba_0),
                     'predict_proba_1': str(predict_proba_1) })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')