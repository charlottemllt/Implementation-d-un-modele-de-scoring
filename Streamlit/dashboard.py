import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle


def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}

def gauge_plot(score):
    fig = go.Figure(go.Indicator(domain = {'x': [0, 1], 'y': [0, 1]},
                                 value = score,
                                 mode = "gauge+number",
                                 title = {'text': "Score client"},
                                 delta = {'reference': 0.91},
                                 gauge = {'axis': {'range': [0, 1]},
                                          'steps' : [
                                            {'range': [0, 0.91], 'color': "whitesmoke"},
                                            {'range': [0.91, 1], 'color': "lightgray"}],
                                          'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.91}}))

    st.plotly_chart(fig)

def gauge(labels=['LOW','HIGH'], colors='jet_r', arrow=1, title=''): 
    # internal functions
    def degree_range(n): 
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang): 
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation
       
    # some sanity checks first
    N = len(labels)
    if arrow > 180: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, 180))
      
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    # begins the plotting   
    fig, ax = plt.subplots()
    ang_range, mid_points = degree_range(N)
    labels = labels[::-1]
    
    # plots the sectors and the arcs
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=1))
    [ax.add_patch(p) for p in patches]

    # set the labels (e.g. 'LOW','MEDIUM',...)
    for mid, lab in zip(mid_points, labels): 
        ax.text(0.34 * np.cos(np.radians(mid)), 0.34 * np.sin(np.radians(mid)),
                lab,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=30,
                rotation = rot_text(mid))

    # set the bottom banner and the title
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    ax.text(0, -0.09, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=90, fontweight='bold')

    # plots the arrow now
    pos = arrow
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.01, head_width=0.03, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    # removes frame and ticks, and makes axis equal and tight
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()


def main():
    st.set_page_config(
        page_title="Tableau de bord",
        page_icon="moneybag",
        layout="wide",
    )

    st.set_option('deprecation.showPyplotGlobalUse', False)

    df_dashboard_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/df_dashboard_lite.csv"
    df = pd.read_csv(df_dashboard_url)
    knn_url = "https://raw.githubusercontent.com/charlottemllt/Implementation-d-un-modele-de-scoring/master/df_knn.csv"
    df_knn = pd.read_csv(knn_url)

    # dashboard title
    st.title("Analyse de solvabilité")


    # Récupération de ID_client
    ID_client = st.sidebar.selectbox("Sélectionnez l'ID client", pd.unique(df['SK_ID_CURR']))
    # Récupération des 10 clients les plus proches

    # creating a single-element container.
    placeholder = st.empty() 

    mean_ext_2 = df['EXT_SOURCE_2'].mean()
    mean_ext_3 = df['EXT_SOURCE_3'].mean()


    # dataframe filter
    df = df[df['SK_ID_CURR'] == ID_client]

    # Take predictions from the API
    session = requests.Session()
    predictions = fetch(session, f"https://api-scoring-credit.herokuapp.com/predict/{ID_client}")
    accord_credit = "Oui" if predictions['retour_prediction'] == '1' else "Non" #✅
    score = float(predictions['predict_proba_1'])

    # Infos descriptives
    GenreCode = df['CODE_GENDER'].mean()
    Genre = 'Femme ♀️' if GenreCode == 1 else 'Homme ♂️'
    Age = int(-df['DAYS_BIRTH'].mean()/365.25)
    Anciennete = int(-df['DAYS_EMPLOYED'].mean()/365.25)
    Voiture = 'Oui' if df['FLAG_OWN_CAR'].mean() == 1 else 'Non'

    # Source externe
    Source2 = df['EXT_SOURCE_2'].mean()
    Source3 = df['EXT_SOURCE_3'].mean()


    # Affichage
    kpi1, kpi2 = st.sidebar.columns(2)
    kpi1.metric(label="Score", value=score)
    kpi2.metric(label="Crédit Accordé", value=accord_credit,)

    st.sidebar.metric(label="Genre", value=Genre)
    st.sidebar.metric(label="Âge", value=f"{Age} ans")
    st.sidebar.metric(label="Voiture personnelle", value=Voiture)
    st.sidebar.metric(label="Ancienneté emploi", value=f"{Anciennete} ans")
    
    with placeholder.container():
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.metric(label="Source externe 2",
                      value=f"{round(Source2, 2)}",
                      delta=f"{round(Source2 - mean_ext_2, 2)}")

            st.metric(label="Source externe 3",
                      value=f"{round(Source3, 2)}",
                      delta=f"{round(Source3 - mean_ext_3, 2)}")

            #st.markdown("Markdown")
            #fig = px.density_heatmap(data_frame=df, y="age_new", x="marital")
            #st.write(fig)
            
        with fig_col2:
            st.markdown("### Second Chart")
            gauge_plot(score)

        #st.markdown("### Detailed Data View")
        #st.dataframe(df)

if __name__ == '__main__':
    main()


"""yes_color = '#007A00'
            no_color =  '#ED1C24'
            threshold=.5
            st.pyplot(gauge(labels=['Granted', 'Rejected'] ,
                             colors=[yes_color, no_color],
                             arrow=180-score*100*1.8-(50-threshold*100)*1.8,
                             title='\n {:.2%}'.format(score)
                            )
                      )"""