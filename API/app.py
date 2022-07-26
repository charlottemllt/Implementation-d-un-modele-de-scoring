from flask import Flask, render_template, jsonify, request, flash
import joblib
import pandas as pd

model = joblib.load('LGBM.joblib').best_estimator_
df_test = pd.read_csv('df_API_lite.csv')
liste_clients = list(df_test['SK_ID_CURR'])

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html")


@app.route('/AllClients/')
def predict():
    """
    Returns
    liste des clients dans le fichier
    """
    return jsonify({'model': 'LGBM',
                    'liste_ID' : list(liste_clients)})


@app.route('/predict/', methods=["POST", "GET"])
def predict_get():
    id = int(request.form['ID_client'])
    seuil = 0.91
    if id in liste_clients:
        X = df_test[df_test['SK_ID_CURR'] == id]
        X.drop('SK_ID_CURR', axis=1, inplace=True)
        predict_ = model.predict_proba(X)
        if predict_[:, 1] >= seuil:
            predict = 'Crédit accordé'
        else:
            predict = 'Crédit non accordé'
    else:
        predict = 'Client inconnu'
    return jsonify({ 'retour_prediction' : str(predict)})

# def predict_result(id):
#     seuil=0.91
#     X = df_test[df_test['SK_ID_CURR'] == id]
#     if X.shape == 0:
#         raise ValueError('{!r} is not a valid ID.'.format(int))
#     else:
#         return 1 if model.predict_proba(X)[1] > seuil else 0

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')