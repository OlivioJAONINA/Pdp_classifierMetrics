import os
import streamlit as st
import pandas as pd
import joblib
import mlflow.pyfunc
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from combat.pycombat import pycombat
from pathlib import Path
from scripts.processing import transform_new_data
pipeline_path = Path(__file__).resolve().parent.parent / 'data/processed/pipeline.json'

sick_model_path = os.path.join(os.path.dirname(__file__), '../models/best_sick_model/best_model.pkl')
sick_model = joblib.load(sick_model_path)
# Interface Streamlit
st.title("Application de prédiction")

# Upload de nouvelles données
uploaded_file = st.file_uploader("Télécharger de nouvelles données", type=['csv'])
# Lecture du fichier le plus récent
raw_dir = Path('/app/data/raw')
raw_file = max(raw_dir.glob("*.csv"), key=lambda f: f.stat().st_mtime)
df_old = pd.read_csv(raw_file)
df_old_subset = df_old.iloc[:, 1:-2]



if uploaded_file:
    data_new = pd.read_csv(uploaded_file)
    num_rows = data_new.shape[0]
    df = pd.concat([df_old_subset, data_new], ignore_index=True)

    # Prétraitement avec le scaler
    processed_data = transform_new_data(df, pipeline_path)
    
    # Prédiction
    predictions = sick_model.predict(processed_data[-num_rows:])
    
    st.write("Résultats des prédictions:")
    df_predictions = pd.DataFrame({'ID3': data_new['ID3'].values, 'Prediction': predictions[-num_rows:]})

    st.write(df_predictions)
