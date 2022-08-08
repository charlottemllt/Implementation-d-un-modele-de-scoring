import pandas as pd
import streamlit as st  # 🎈 data web app development
import requests
import plotly.graph_objects as go
import joblib
from streamlit_shap import st_shap
import shap
import plotly.express as px
import numpy as np
import matplotlib
from collections import Counter

best_model = joblib.load('LGBM.joblib').best_estimator_

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def feature_engineering(df):
    new_df = pd.DataFrame()
    #colonnes_non_modif = ['SK_ID_CURR', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_CREDIT',
     #                     'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS', 'INCOME_PER_PERSON']
    new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x : -x/365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : -x/365.25)
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    new_df['NAME_EDUCATION_TYPE_Highereducation'] = df['NAME_EDUCATION_TYPE_Highereducation'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df

def get_english_var(var_fr):
    liste_var_en = ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS',
                    'INCOME_PER_PERSON', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation', 'AMT_GOODS_PRICE', 'AMT_CREDIT',
                    'PAYMENT_RATE', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    liste_var_fr = ['ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux', 'Nombre de personnes dans la famille',
                    'Revenus par personne', 'Voiture personnelle', 'Education secondaire',
                    'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit', 'Fréquence de paiement',
                    'Montant des annuités', 'Source externe 2', 'Source externe 3']
    ind = liste_var_fr.index(var_fr)
    var_en = liste_var_en[ind]
    return var_en

def main():
    st.set_page_config(
        page_title="Tableau de bord",
        page_icon="moneybag",
        layout="wide",
    )

    df_dashboard_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/Dashboard/df_dashboard_lite.csv"
    df = pd.read_csv(df_dashboard_url)

    # dashboard title
    st.title("Analyse de solvabilité")

    # Récupération de ID_client
    ID_client = st.sidebar.selectbox("Sélectionnez l'ID client", pd.unique(df['SK_ID_CURR']))

    # dataframe filter
    df_client = df[df['SK_ID_CURR'] == ID_client]
    new_df = feature_engineering(df_client)

    # Take predictions from the API
    session = requests.Session()
    predictions = fetch(session, f"https://api-scoring-credit.herokuapp.com/predict/{ID_client}")
    accord_credit = "Oui" if predictions['retour_prediction'] == '1' else "Non" #✅
    score = float(predictions['predict_proba_1'])

    # Shap values
    explainer = shap.TreeExplainer(best_model)
    df_api_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/Dashboard/df_API_lite.csv"
    df_API = pd.read_csv(df_api_url)
    df_shap = df_API.loc[:, df_API.columns != 'SK_ID_CURR']
    shap_values = explainer.shap_values(df_shap)
    
    # Affichage
    st.sidebar.metric(label="Crédit Accordé", value=accord_credit)

    st.sidebar.header("Informations générales")
    kpi1, kpi2 = st.sidebar.columns(2)
    kpi1.metric(label="Genre", value='Femme' if df_client['CODE_GENDER'].mean() == 1 else 'Homme' ) # ♀️ ♂️
    kpi2.metric(label="Âge", value=f"{int(int(-df_client['DAYS_BIRTH'].mean()/365.25))} ans")
    
    kpi3, kpi4 = st.sidebar.columns(2)
    kpi3.metric(label="Revenus totaux", value=str(df_client['AMT_INCOME_TOTAL'].mean()/1000) + 'k')
    kpi4.metric(label="Ancienneté emploi", value=f"{int(-df_client['DAYS_EMPLOYED'].mean()/365.25)} ans")

    st.sidebar.metric(label="Nombre de personnes dans la famille", value=int(df_client['CNT_FAM_MEMBERS'].mean()))

    tab1, tab2, tab3 = st.tabs(['Score client', 'Explication du score', 'Comparaison aux autres clients'])
    with tab1:
        st.header('Score de solvabilité')
        if score < 0.91:
            fig = go.Figure(go.Indicator(
                                mode = 'gauge + number',
                                value = score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                delta = {'reference': 0.91},
                                gauge = {'axis': {'range': [0, 1]},
                                        'bar': {'color': 'red'},
                                        'steps' : [{'range': [0, 0.91], 'color': "lightgrey"},
                                                    {'range': [0.91, 1], 'color': "grey"}],
                                        'threshold' : {'line': {'color': 'green', 'width': 4}, 'thickness': 0.75, 'value': 0.91}}
                            ))
        else:
            fig = go.Figure(go.Indicator(
                                mode = 'gauge + number',
                                value = score,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                delta = {'reference': 0.91},
                                gauge = {'axis': {'range': [0, 1]},
                                        'bar': {'color': 'green'},
                                        'steps' : [{'range': [0, 0.91], 'color': "grey"},
                                                    {'range': [0.91, 1], 'color': "lightgrey"}],
                                        'threshold' : {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': 0.91}}
                            ))
        st.plotly_chart(fig)
    
        # Interprétation pour l'individu choisi
        st.header("Impact des variables sur le score pour le client " + str(ID_client))
        id = df_API[df_API['SK_ID_CURR'] == ID_client].index
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][id, :], df_shap.iloc[id, :], link='logit'))
        st.write('Les variables en rose ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en bleu ont contribué à refuser le crédit (donc à diminuer le score)')
    
    with tab2:
        # Interprétation pour l'ensemble des clients
        st.header("Impact des variables pour l'ensemble des clients")
        st.write('Les variables en rose ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en bleu ont contribué à refuser le crédit (donc à diminuer le score)')
        st_shap(shap.summary_plot(shap_values, df_shap))

    with tab3:
        st.header("Comparaison aux autres clients")
        categ = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation']
        
        col1, col2 = st.columns(2)
        with col1:
            liste_variables1 = ['Revenus par personne', 'ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Voiture personnelle', 'Education secondaire',
                                'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
                    
            variable1 = st.selectbox("Sélectionnez la première variable à afficher", liste_variables1, key=1)
            var_en1 = get_english_var(variable1)
            if var_en1 in categ:
                var1_cat = 1
            else:
                var1_cat = 0
        
        with col2:
            liste_variables2 = ['Ancienneté de l\'emploi', 'ID_Client', 'Genre', 'Âge', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Revenus par personne', 'Voiture personnelle',
                                'Education secondaire', 'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
            variable2 = st.selectbox("Sélectionnez la seconde variable à afficher", liste_variables2, key=2)
            var_en2 = get_english_var(variable2)
            if var_en2 in categ:
                var2_cat = 1
            else:
                var2_cat = 0
        
        df_comp = pd.read_csv('https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/Dashboard/df_comp_lite.csv')
        df_comp = feature_engineering(df_comp)
        if variable1 == variable2:
            df_comp = df_comp[[var_en1, 'TARGET', 'Score']].dropna()
        else:   
            df_comp = df_comp[[var_en1, var_en2, 'TARGET', 'Score']].dropna()
        
        col1_, col2_ = st.columns(2)
        with col1_:
            if var1_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig1 = px.histogram(df_comp, x=var_en1, color='TARGET', marginal=marg, nbins=50)
            if var1_cat == 0:
                fig1.add_vline(x=new_df[var_en1].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(ID_client))
            fig1.update_layout(barmode='overlay', title={'text': variable1, 'x': 0.5, 'xanchor': 'center'})
            fig1.update_traces(opacity=0.75)
            st.plotly_chart(fig1, use_container_width=True)
        with col2_:
            if var2_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig2 = px.histogram(df_comp, x=var_en2, color='TARGET', marginal=marg, nbins=50)
            if var2_cat == 0:
                fig2.add_vline(x=new_df[var_en2].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(ID_client))
            fig2.update_layout(barmode='overlay', title={'text': variable2, 'x': 0.5, 'xanchor': 'center'})
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)
        
        if ((var1_cat + var2_cat) == 0) or (var1_cat == 1 and var2_cat == 0):
            scat = px.scatter(df_comp, x=var_en2, y=var_en1, color='Score', opacity=0.75,
                              color_continuous_scale=[(0.0, 'darkred'), (0.5, 'red'),
                                                      (0.5, 'red'), (0.7, 'orange'),
                                                      (0.7, 'orange'), (0.91, 'yellow'),
                                                      (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[var_en2], y=new_df[var_en1], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(ID_client)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(scat, use_container_width=True)
        elif (var1_cat + var2_cat) == 2:
            table = np.round(pd.pivot_table(df_comp, values='Score', index=[var_en1],
                                            columns=[var_en2], aggfunc=np.mean),
                             2) 
            fig = px.imshow(table, text_auto=True, color_continuous_scale='Blues')
            #[(0.0, 'red'), (0.5, 'orange'), (0.5, 'orange'), (0.7, 'yellow'), (0.7, 'yellow'), (0.91, 'lime'), (0.91, 'lime'),  (1.0, 'green')]
            st.write(fig)
        else:
            scat = px.scatter(df_comp, x=var_en1, y=var_en2, color='Score', opacity=0.75,
            color_continuous_scale=[(0.0, 'darkred'), (0.5, 'red'),
                                    (0.5, 'red'), (0.7, 'orange'),
                                    (0.7, 'orange'), (0.91, 'yellow'),
                                    (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[var_en1], y=new_df[var_en2], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(ID_client)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(scat, use_container_width=True)


if __name__ == '__main__':
    main()