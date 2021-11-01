import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize
import pickle
from sklearn.neighbors import KDTree
import plotly.graph_objects as go


def undummify(df, prefix_sep="_"):
    col_to_undummify = []
    for col in df.columns:
        print(df[col].value_counts().keys())
        print(len(df[col].value_counts().keys()))

        if len(df[col].value_counts().keys()) == 2:
            print(type(df[col].iloc[0]))
            if isinstance(df[col].iloc[0], float):
                end_col = col.split(prefix_sep)[-1]
                if (not end_col.isupper() and not end_col.isdigit()) or (end_col == "F") or (end_col == "M"):
                    print(col)
                    print("_".join(col.split(prefix_sep)[:-1]))
                    print(col_to_undummify)
                    col_to_undummify.append(col)
    cols2collapse = {
        "_".join(item.split(prefix_sep)[:-1]): (prefix_sep in item) for item in col_to_undummify}
    for col in df.columns:
        if col not in col_to_undummify:
            cols2collapse[col] = False
    print(cols2collapse)
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep)[-1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


def main():
    # Colonne présentant les clients
    st.title("Algorithme d'évaluation de Solvabilité d'un Client")

    filename_df_training = "dataframe_entraînement_id.csv"
    filename_df_raw = "dataframe_training_undummify_id.csv"
    filename_df_norm = "dataframe_training_norm_id.csv"
    # data_load_state = st.text('Loading data...')

    data_training_df = pd.read_csv(filename_df_training, sep=";", index_col='SK_ID_CURR', nrows=1000, header=0)
    data_train_undu_df = pd.read_csv(filename_df_raw, sep=",", index_col='SK_ID_CURR', nrows=1000, header=0)
    dataframe_train_norm_id = pd.read_csv(filename_df_norm, sep=",", index_col='SK_ID_CURR', nrows=1000, header=0)

    todo = 0
    if todo:
        """ Creating Undummified DataFrame from dataframe_entraînement_id.csv"""
        print(data_training_df.head())
        data_train_undu_df = undummify(data_training_df)
        print(data_train_undu_df.head())
        print(data_train_undu_df.columns)
        data_train_undu_df.to_csv("dataframe_training_undummify_id.csv")
        """"""

    todo = 0
    if todo:
        """ Creating Normalized DataFrame from dataframe_entraînement_id.csv"""
        print(data_training_df.head())
        data_train_norm_df = data_training_df.copy()
        x_norm = normalize(data_train_norm_df.values)
        dataframe_train_norm_id = pd.DataFrame(data=x_norm, index=data_train_norm_df.index,
                                               columns=data_train_norm_df.columns)
        print(dataframe_train_norm_id.head())
        print(dataframe_train_norm_id.columns)
        dataframe_train_norm_id.to_csv("dataframe_training_norm_id.csv")
        """"""

    # data_load_state.text('Loading data...done!')
    # """ Selection du client """
    id_client = st.selectbox("Selectionnez un ID de Client",
                             options=data_training_df.index.tolist())

    # """ Informations du client actuel """
    st.sidebar.table(data_train_undu_df.loc[id_client][:8].rename({"NAME_CONTRACT_TYPE": 'Type de Contrat',
                                                                   'CODE_GENDER': "Sexe", "NAME_TYPE_SUITE": "Groupe",
                                                                   'NAME_INCOME_TYPE': "Source de Revenu",
                                                                   'NAME_EDUCATION_TYPE': 'Education',
                                                                   'NAME_FAMILY_STATUS': 'Statut Familial',
                                                                   'NAME_HOUSING_TYPE': 'Type de Logement',
                                                                   'OCCUPATION_TYPE': 'Emploi'}))

    # """ Recherche des clients les plus proches """
    _, _, filenames = next(os.walk(os.getcwd()))
    found_tree = False
    for filename in filenames:
        if "tree" in filename:
            found_tree = True
            with open('filename', 'rb') as file:
                tree = pickle.load(file)
    if not found_tree:
        tree = KDTree(dataframe_train_norm_id.values, leaf_size=40)

    _, ind = tree.query(dataframe_train_norm_id.loc[id_client].values.reshape(1, -1), k=10)

    all_clients = st.checkbox('Comparer à tout les clients')
    if all_clients:
        neiborgs_df = data_training_df
    else:
        neiborgs_df = data_training_df.iloc[ind[0], :]

    # """ Chargement du model"""
    choosen_col = ['TOTALAREA_MODE', 'AMT_CREDIT', 'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'NONLIVINGAREA_MODE', 'LIVINGAREA_MODE', 'ORGANIZATION_TYPE_Transporttype2', 'NONLIVINGAREA_AVG', 'BASEMENTAREA_AVG', 'OBS_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT', 'BASEMENTAREA_MEDI', 'FLOORSMAX_MEDI', 'LIVINGAPARTMENTS_MODE', 'AMT_REQ_CREDIT_BUREAU_MON', 'LANDAREA_MODE', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'BASEMENTAREA_MODE', 'REGION_POPULATION_RELATIVE', 'LIVINGAREA_AVG', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OWN_CAR_AGE', 'ORGANIZATION_TYPE_Security', 'COMMONAREA_AVG', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_ANNUITY', 'DAYS_REGISTRATION', 'HOUR_APPR_PROCESS_START', 'AMT_GOODS_PRICE', 'COMMONAREA_MEDI', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'COMMONAREA_MODE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_ID_PUBLISH',  'EXT_SOURCE_3']
    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    # """ Calcul du score des voisins"""
    result_neigbors = model.predict_proba(neiborgs_df[choosen_col].values, pred_contrib=False)
    result_mean_solvable = np.mean(result_neigbors, axis=1)[0]
    predict_contrib_nei = model.predict_proba(neiborgs_df[choosen_col].values, pred_contrib=True)
    # print(result_neigbors)
    print(predict_contrib_nei)      # Seules les 40 premières valeurs sont utiles

    # """ Calcul du score du client"""
    result_prob = model.predict_proba(data_training_df[choosen_col].loc[id_client].values.reshape(1, -1), pred_contrib=False)
    result_client_solvable = result_prob[0][0]
    predict_contrib_id = model.predict_proba(data_training_df[choosen_col].loc[id_client].values.reshape(1, -1),
                                             pred_contrib=True)[0]
    print(predict_contrib_id)      # Seules les 40 premières valeurs sont utiles

    # """ Affichage Resultat"""
    colu_1, colu_2 = st.columns([1, 3])
    with colu_1:
        st.metric(label="Probabilité de Solvabilité", value=str(np.round(result_client_solvable,3)),
                  delta=str(np.round(result_mean_solvable - result_client_solvable, 3)))
    with colu_2:
        fig_2 = go.Figure(go.Indicator(mode="number+gauge+delta",
                                       value=result_client_solvable*100,
                                       domain={'x': [0, 1], 'y': [0, 1]},
                                       delta={'reference': np.round(result_mean_solvable - result_client_solvable, 3)*100,
                                              'increasing': {'color': 'red'},
                                                'decreasing' : {'color' : 'green'},
                                                'position': "top"},
                                       title={'text':"<b>En %</b><br><span style='color: gray; font-size:0.8em'></span>", 'font': {"size": 14}},
                                       gauge={'shape': "bullet",
                                                'axis': {'range': [None, 100]},
                                                'threshold': {
                                                    'line': {'color': "white", 'width': 3},
                                                    'thickness': 0.75, 'value': result_client_solvable*100},
                                                'bgcolor': "white",
                                                'steps': [{'range': [0, 85], 'color': "red"},
                                                          {'range': [85, 92], 'color': "orange"},
                                                          {'range': [92, 100], 'color': "lightgreen"}],
                                                'bar': {'color': "darkblue"}}))
        fig_2.update_layout(height=200)

        st.plotly_chart(fig_2)
    # """ Comparaison aux clients les plus proches"""

    # """ Choix du nombre de features"""
    n_features = st.slider('Combien de caractéristiques à comparer?', 1, 40, 7)

    # """ Recherche des colonnes les plus significatives """
    mean_nei = np.mean(predict_contrib_nei, axis=0)[:-1]   # On enlève la dernière valeur
    ind = np.argpartition(np.abs(mean_nei), -n_features)[-n_features:]    # On prends les n features les plus grandes
    col_nei_features_max_mean = [choosen_col[curr_ind] for curr_ind in ind[np.argsort(np.abs(mean_nei)[ind])]]
    col_nei_features_max_mean.reverse()     # Les plus grands en premier

    # """ Recuperation des features importances pour ces colonnes"""
    features_importance_df = pd.DataFrame(index=col_nei_features_max_mean)
    features_importance_df['Clients Proches'] = [mean_nei[choosen_col.index(col_features)] for col_features in col_nei_features_max_mean]
    features_importance_df['Client Actuel'] = [predict_contrib_id[choosen_col.index(col_features)] for col_features in col_nei_features_max_mean]
    print(features_importance_df)
    # features_importance_df.sort_values(by="Client Actuel", inplace=True)
    print(len(col_nei_features_max_mean))

    col1, col2 = st.columns(2)
    # with col1:
    features_importance_df["Color Proches"] = np.where(features_importance_df["Clients Proches"] < 0, 'red', 'green')

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Clients Proches',
               x=features_importance_df.index,
               y=features_importance_df['Clients Proches'],
               marker_color=features_importance_df["Color Proches"]))
    fig.update_layout(title="Principaux determinants du score des clients proches",
                      font_family="Courier New",
                      font_color="white",
                      title_font_family="Times New Roman",
                      title_font_color="white",
                      legend_title_font_color="green",
                      barmode='stack')
    st.plotly_chart(fig)

    #   Ancienne version
    # st.header("Principaux determinants du score des clients proches")
    # st.bar_chart(data=features_importance_df['Clients Proches'], width=0, height=0, use_container_width=True)
    features_importance_df["Color Actuel"] = np.where(features_importance_df["Client Actuel"] < 0, 'red', 'green')

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Client Actuel',
               x=features_importance_df.index,
               y=features_importance_df['Client Actuel'],
               marker_color=features_importance_df["Color Actuel"]))
    fig.update_layout(title="Principaux determinants du score du client actuel",
                      font_family="Courier New",
                      font_color="white",
                      title_font_family="Times New Roman",
                      title_font_color="white",
                      legend_title_font_color="green",
                      barmode='stack')
    st.plotly_chart(fig)
    # with col2:
    # st.header("Principaux determinants du score du client actuel")
    # st.bar_chart(data=features_importance_df['Client Actuel'], width=0, height=0, use_container_width=True)

    return 0


if __name__ == '__main__':
    main()
