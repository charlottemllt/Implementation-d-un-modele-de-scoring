import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development
import requests
import plotly.graph_objects as go
import joblib
from streamlit_shap import st_shap
import shap
import plotly.express as px
import numpy as np
import matplotlib

best_model = joblib.load('LGBM.joblib').best_estimator_

def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def feature_engineering(df):
    new_df = pd.DataFrame()
    colonnes_non_modif = ['SK_ID_CURR', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_CREDIT']
    for i in range(len(colonnes_non_modif)):
        new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x : -x/365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : -x/365.25)
    new_df['NAME_FAMILY_STATUS_Married'] = df['NAME_FAMILY_STATUS_Married'].apply(lambda x: 'Marié(é)' if x == 1 else 'Célibataire')
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    new_df['NAME_EDUCATION_TYPE_Highereducation'] = df['NAME_EDUCATION_TYPE_Highereducation'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df

def main():
    st.set_page_config(
        page_title="Tableau de bord",
        page_icon="moneybag",
        layout="wide",
    )

    df_dashboard_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/df_dashboard_lite.csv"
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
    
    # Affichage
    st.sidebar.metric(label="Crédit Accordé", value=accord_credit,)

    st.sidebar.header("Informations générales")
    kpi1, kpi2 = st.sidebar.columns(2)
    kpi1.metric(label="Genre", value='Femme' if df_client['CODE_GENDER'].mean() == 1 else 'Homme' ) # ♀️ ♂️
    kpi2.metric(label="Âge", value=f"{int(int(-df_client['DAYS_BIRTH'].mean()/365.25))} ans")
    
    kpi3, kpi4 = st.sidebar.columns(2)
    kpi3.metric(label="Voiture personnelle", value='Oui' if df_client['FLAG_OWN_CAR'].mean() == 1 else 'Non')
    kpi4.metric(label="Marié(e)", value='Oui' if df_client['NAME_FAMILY_STATUS_Married'].mean() == 1 else 'Non')

    kpi5, kpi6 = st.sidebar.columns(2)
    kpi5.metric(label="Education secondaire", value='Oui' if df_client['NAME_EDUCATION_TYPE_Highereducation'].mean() == 1 else 'Non')
    kpi6.metric(label="Ancienneté emploi", value=f"{int(-df_client['DAYS_EMPLOYED'].mean()/365.25)} ans")

    tab1, tab2, tab3 = st.tabs(["Score client", "Comparaison aux autres clients", 'Explication du score'])
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

    with tab2:
        st.header("Comparaison aux autres clients")
        categ = ['CODE_GENDER', 'NAME_FAMILY_STATUS_Married', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation']
        col1, col2 = st.columns(2)
        with col1:
            liste_variables1 = ['CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_FAMILY_STATUS_Married', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation',
                                'AMT_GOODS_PRICE', 'AMT_CREDIT', 'PAYMENT_RATE', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
            variable1 = st.selectbox("Sélectionnez la première variable à afficher", liste_variables1, key=1)
            if variable1 in categ:
                var1_cat = 1
            else:
                var1_cat = 0
        
        with col2:
            liste_variables2 = ['DAYS_BIRTH', 'CODE_GENDER', 'DAYS_EMPLOYED', 'NAME_FAMILY_STATUS_Married', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation',
                                'AMT_GOODS_PRICE', 'AMT_CREDIT', 'PAYMENT_RATE', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
            variable2 = st.selectbox("Sélectionnez la seconde variable à afficher", liste_variables2, key=2)
            if variable2 in categ:
                var2_cat = 1
            else:
                var2_cat = 0
        
        df_comp = pd.read_csv('df_dashboard_comp.csv')
        if variable1 == variable2:
            df_comp = df_comp[[variable1, 'TARGET', 'Score']].dropna()
        else:   
            df_comp = df_comp[[variable1, variable2, 'TARGET', 'Score']].dropna()
        
        col1_, col2_ = st.columns(2)
        with col1_:
            if var1_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig1 = px.histogram(df_comp, x=variable1, color='TARGET', marginal=marg, nbins=50)
            if var1_cat == 0:
                fig1.add_vline(x=new_df[variable1].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(ID_client))
            fig1.update_layout(barmode='overlay')
            fig1.update_traces(opacity=0.75)
            st.plotly_chart(fig1, use_container_width=True)
        with col2_:
            if var2_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig2 = px.histogram(df_comp, x=variable2, color='TARGET', marginal=marg, nbins=50)
            if var2_cat == 0:
                fig2.add_vline(x=new_df[variable2].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(ID_client))
            fig2.update_layout(barmode='overlay')
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)
        
        if ((var1_cat + var2_cat) == 0) or (var1_cat == 1 and var2_cat == 0):
            scat = px.scatter(df_comp, x=variable2, y=variable1, color='Score', opacity=0.75,
                              color_continuous_scale=[(0.0, 'darkred'),   (0.5, 'red'),
                                                      (0.5, 'red'), (0.7, 'orange'),
                                                      (0.7, 'orange'), (0.91, 'yellow'),
                                                      (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[variable2], y=new_df[variable1], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(ID_client)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(scat, use_container_width=True)
        elif (var1_cat + var2_cat) == 2:
            table = np.round(pd.pivot_table(df_comp, values='Score', index=[variable1],
                                            columns=[variable2], aggfunc=np.mean),
                             2) 
            fig = px.imshow(table, text_auto=True, color_continuous_scale='Blues')
            #[(0.0, 'red'), (0.5, 'orange'), (0.5, 'orange'), (0.7, 'yellow'), (0.7, 'yellow'), (0.91, 'lime'), (0.91, 'lime'),  (1.0, 'green')]
            st.write(fig)
        else:
            scat = px.scatter(df_comp, x=variable1, y=variable2, color='Score', opacity=0.75,
            color_continuous_scale=[(0.0, 'darkred'),   (0.5, 'red'),
                                    (0.5, 'red'), (0.7, 'orange'),
                                    (0.7, 'orange'), (0.91, 'yellow'),
                                    (0.91, 'green'),  (1.0, 'green')])
            scat.add_trace(go.Scatter(x=new_df[variable1], y=new_df[variable2], mode='markers',
                                      marker=dict(size=16, color='#8f00ff'), opacity=0.99, name='Client ' + str(ID_client)))
            scat.update_layout(legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(scat, use_container_width=True)


    with tab3:
        # Interprétation pour l'ensemble des clients
        explainer = shap.TreeExplainer(best_model)
        df_api_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/API/df_API_lite.csv"
        df_API = pd.read_csv(df_api_url)
        df_shap = df_API.loc[:, df_API.columns != 'SK_ID_CURR']
        shap_values = explainer.shap_values(df_shap)
        st.header("Impact des variables pour l'ensemble des clients")
        st_shap(shap.summary_plot(shap_values, df_shap))

        # Interprétation pour l'individu choisi
        st.header("Impact des variables sur le score pour le client " + str(ID_client))
        id = df_API[df_API['SK_ID_CURR'] == ID_client].index
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][id, :], df_shap.iloc[id, :], link='logit'))

if __name__ == '__main__':
    main()