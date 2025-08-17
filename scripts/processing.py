import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from combat.pycombat import pycombat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import pickle
import json


def evaluate_models(X, y, class_weight=None, n_splits=5, random_state=42, svm_kernels=None, poly_degrees=None):
    """
    Standardise les données, applique une validation croisée StratifiedKFold et évalue plusieurs modèles.
    
    Paramètres :
        - X : np.array ou pd.DataFrame, les caractéristiques.
        - y : np.array ou pd.Series, les labels.
        - class_weight : str ou dict, poids des classes ('balanced' ou un dictionnaire de poids).
        - n_splits : int, nombre de folds pour la validation croisée.
        - random_state : int, graine aléatoire pour reproductibilité.
        - svm_kernels : list, liste des noyaux SVM à tester (ex: ['linear', 'rbf', 'poly', 'sigmoid']).
        - poly_degrees : list, liste des degrés à tester pour le noyau polynomial.

    Retourne :
        - DataFrame contenant les scores globaux pour chaque modèle.
        - Dictionnaire contenant classification_report et confusion_matrix pour chaque modèle.
    """

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Définition des modèles de base
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', class_weight=class_weight, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight=class_weight, random_state=random_state)
    }

    # Définition des noyaux SVM par défaut
    if svm_kernels is None:
        svm_kernels = ['linear', 'rbf']  # Valeurs par défaut

    # Ajout des modèles SVM avec les noyaux demandés
    for kernel in svm_kernels:
        if kernel == "poly" and poly_degrees:  # Gérer plusieurs degrés pour poly
            for d in poly_degrees:
                models[f"SVM (poly, degree={d})"] = SVC(kernel="poly", degree=d, class_weight=class_weight, random_state=random_state)
            # Ajout de SVM poly avec C=1000
                models[f"SVM (poly, degree={d}, C=1000)"] = SVC(kernel="poly", degree=d, C=1000, class_weight=class_weight, random_state=random_state)
        else:
            models[f"SVM ({kernel})"] = SVC(kernel=kernel, class_weight=class_weight, random_state=random_state)

    # Création de la validation croisée stratifiée
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    detailed_reports = {}

    # Boucle sur les modèles
    for name, model in models.items():
        acc_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
        all_y_true, all_y_pred = [], []

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Sauvegarde des métriques pour chaque fold
            acc_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

            # Stockage des prédictions réelles et prédites pour un rapport final
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        # Moyenne des scores sur les folds
        results.append({
            "Modèle": name,
            "Accuracy": np.mean(acc_scores),
            "Precision": np.mean(precision_scores),
            "Recall": np.mean(recall_scores),
            "F1-score": np.mean(f1_scores)
        })

        # Rapport détaillé
        detailed_reports[name] = {
            "classification_report": classification_report(all_y_true, all_y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(all_y_true, all_y_pred)
        }

    # Convertir en DataFrame pour affichage structuré
    return pd.DataFrame(results).sort_values(by="F1-score", ascending=False)


'''


def preprocess_data(raw_data_dir,processed_dir, corr_trait=True, corr_threshold=0.9, 
                    apply_pca=True, pca_variance_threshold=0.95, combat=True):
    """
    Prend en entrée une matrice X et un vecteur batch, applique un traitement incluant :
    - Suppression des colonnes fortement corrélées (optionnel)
    - Réduction de dimension par PCA (optionnel)
    - Correction de l'effet batch avec pyComBat

    Paramètres :
        - X : pd.DataFrame ou np.array, matrice des caractéristiques.
        - batch : pd.Series ou np.array, vecteur indiquant l'appartenance à un batch.
        - corr_trait : bool, active ou désactive la suppression des colonnes corrélées.
        - corr_threshold : float, seuil de corrélation au-dessus duquel une variable est supprimée.
        - apply_pca : bool, active ou désactive l'application de PCA.
        - pca_variance_threshold : float, proportion minimale de variance à conserver après PCA.

    Retourne :
        - X_traited : np.array, matrice transformée après toutes les étapes.
    """
    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_dir)
    
    # Trouver le fichier le plus récent
    input_file = max(raw_dir.glob('*.csv'), key=lambda f: f.stat().st_mtime)
    
    # Lire et traiter les données
    df = pd.read_csv(input_file)
    
    X = np.array(df.iloc[:,:-1])
    batch = np.array(df.iloc[:,-1])
    # Conversion en DataFrame si nécessaire
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # 1️ Suppression des colonnes fortement corrélées 
    if corr_trait:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
        X = X.drop(columns=to_drop)
        print(f"---->>> Colonnes supprimées pour corrélation élevée : {to_drop}")

    # 2️ Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3️ Réduction de dimension avec PCA 
    if apply_pca:
        pca = PCA()
        #X_pca = pca.fit_transform(X_scaled)
        
        # Trouver le nombre de composantes qui expliquent au moins `pca_variance_threshold` de la variance
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(explained_variance_ratio >= pca_variance_threshold) + 1
        print(f" Nombre de composantes PCA conservées : {num_components} / {X.shape[1]}")
        #print(np.max(explained_variance_ratio))
        
        # Appliquer PCA avec le bon nombre de composantes
        pca = PCA(n_components=num_components)
        X_scaled = pca.fit_transform(X_scaled)
    if combat:
        feature_names = [f"var_{i}" for i in range(X_scaled.shape[1])]
        data = pd.DataFrame(X_scaled, columns=feature_names)
        X_scaled = pycombat(data.T,batch).T

    return X_scaled
    

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from pycombat import pycombat  # Adapter selon ton installation (combat.py ou pycombat)
'''

def preprocess_data(raw_data_dir, processed_dir,
                    corr_trait=True, corr_threshold=0.9,
                    apply_pca=True, pca_variance_threshold=0.95,
                    combat=True):
    """
    Prétraitement d'un fichier de données tabulaires pour l'entraînement d'un modèle de classification.

    Étapes :
        - Suppression des colonnes fortement corrélées (optionnelle)
        - Standardisation (obligatoire)
        - Réduction de dimension par PCA (optionnelle)
        - Correction batch avec pyComBat (optionnelle)

    Sauvegardes :
        - Données transformées (X_processed.npy)
        - Pipeline (pipeline.pkl) contenant :
            - scaler
            - pca
            - colonnes supprimées
            - indicateur combat
            - seuil de corrélation

    Paramètres :
        raw_data_dir : str | dossier contenant les fichiers CSV bruts
        processed_dir : str | dossier de sauvegarde des résultats
        corr_trait : bool | appliquer ou non la suppression de colonnes corrélées
        corr_threshold : float | seuil de corrélation
        apply_pca : bool | appliquer ou non la PCA
        pca_variance_threshold : float | pourcentage minimal de variance à conserver via PCA
        combat : bool | appliquer ou non la correction batch

    Retour :
        X_scaled : np.ndarray | données transformées
        pipeline : dict | objets de transformation sauvegardés
    """

    raw_dir = Path(raw_data_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Lecture du fichier le plus récent
    input_file = max(raw_dir.glob("*.csv"), key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(input_file)

    # Séparation données / batch
    X = df.iloc[:, 1:-3]
    
    #batch = df.iloc[:, -1].values
    batch = df[["Batch"]].values.flatten()  # Assurez-vous que c'est un vecteur 1D

    # Initialisation du pipeline
    pipeline = {
        "columns_removed": [],
        "pca": None,
        "pca_n_components": None,
        "combat": combat,
        "corr_threshold": corr_threshold
    }

    # 1️⃣ Suppression des colonnes fortement corrélées
    if corr_trait:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
        X = X.drop(columns=to_drop)
        pipeline["columns_removed"] = to_drop
        print(f"---->>> Colonnes supprimées pour corrélation > {corr_threshold} : {to_drop}")

    # 2️⃣ Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # 3️⃣ PCA (optionnelle)
    if apply_pca:
        pca_tmp = PCA().fit(X_scaled)
        cum_var = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_components = np.argmax(cum_var >= pca_variance_threshold) + 1
        print(f"---->>> Composantes PCA conservées : {n_components}/{X.shape[1]}")
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        pipeline["pca"] = True
        pipeline["pca_n_components"] = n_components

    # 4️⃣ Correction batch (optionnelle)
    if combat:
        feature_names = [f"var_{i}" for i in range(X_scaled.shape[1])]
        data = pd.DataFrame(X_scaled, columns=feature_names)
        X_scaled = pycombat(data.T, batch).T
        print(f"---->>> Correction batch appliquée (ComBat)")
        

    # Sauvegarde des données traitées
    # Sauvegarde des données traitées en .csv
    df_processed = pd.DataFrame(X_scaled, columns=[f"var_{i}" for i in range(X_scaled.shape[1])])
    #df_final = pd.concat([df_processed, df[df.columns[-3:],"ID3"]], axis=1)
    cols_to_keep = list(df.columns[-3:]) + ['ID3']
    df_final = pd.concat([df_processed, df[cols_to_keep]], axis=1)
    df_final["sick"] = df['Class_simple'].apply(lambda x: 0 if x == 0 else 1)
    df_final.to_csv(processed_dir / "data_processed.csv", index=False)

    # Sauvegarde du pipeline complet
    #with open(processed_dir / "pipeline.pkl", "wb") as f:
        #pickle.dump(pipeline, f)
    #json.dump(pipeline, open(processed_dir / "pipeline.json", "w"), indent=4)
    
    with open(processed_dir / "pipeline.json", "w") as f:
        json.dump(pipeline, f, indent=4,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    print(f"✅ Données transformées sauvegardées dans : {processed_dir / 'X_processed.csv'}")
    print(f"✅ Pipeline de transformation sauvegardé dans : {processed_dir / 'pipeline.pkl'}")

#    return X_scaled, pipeline

    return str(Path("/opt/airflow/data/processed")) 


'''
def transform_new_data(X_new, pipeline):
    """
    Applique les étapes du pipeline de prétraitement sur de nouvelles données.

    Paramètres :
        X_new : pd.DataFrame | nouvelles données à transformer (même structure que les données d'entraînement AVANT prétraitement)
        pipeline : dict | dictionnaire contenant :
            - 'columns_removed' : list des colonnes à supprimer
            - 'scaler' : objet StandardScaler
            - 'pca' : objet PCA ou None
            - 'combat' : bool (non utilisé ici)
            - 'corr_threshold' : float (info seulement)

    Retour :
        X_transformed : np.ndarray | nouvelles données transformées prêtes pour l'inférence
    """

    # 1️⃣ Suppression des colonnes corrélées (même que lors du fit)
    if pipeline["columns_removed"]:
        X_new = X_new.drop(columns=pipeline["columns_removed"], errors="ignore")

    # 2️⃣ Vérification que les colonnes restantes sont identiques
    if pipeline["scaler"] is None:
        raise ValueError("Le pipeline ne contient pas d'objet scaler valide.")
    
    # 3️⃣ Standardisation
    X_transformed = pipeline["scaler"].transform(X_new)

    # 4️⃣ PCA (si utilisée)
    if pipeline["pca"] is not None:
        X_transformed = pipeline["pca"].transform(X_transformed)

    # 5️⃣ Pas de ComBat ici, car le batch d’inférence n’est souvent pas connu.

    return X_transformed
'''

import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def transform_new_data(data_new, pipeline_path):
    
    """
    Applique les étapes du pipeline de prétraitement sur de nouvelles données.

    Paramètres :
        X_new : pd.DataFrame | nouvelles données à transformer
        pipeline_path : str | chemin vers le fichier JSON contenant la configuration du pipeline

    Retour :
        X_transformed : np.ndarray | nouvelles données transformées prêtes pour l'inférence
    """
    batch = data_new["Batch"].values.flatten()  # Assurez-vous que c'est un vecteur 1D
    X_new = data_new.iloc[:, 1:-3]
    # 1. Charger la configuration du pipeline depuis le fichier JSON
    with open(pipeline_path, 'r') as f:
        pipeline = json.load(f)

    # 2. Suppression des colonnes corrélées
    if pipeline.get("columns_removed") is not None:
        X_new = X_new.drop(columns=pipeline["columns_removed"], errors="ignore")


    # Ici vous devriez charger votre scaler entraîné (exemple simplifié)
    # En pratique, vous auriez sauvegardé le scaler avec joblib/pickle
    scaler = StandardScaler()  # Remplacez par votre vrai scaler chargé
    
    X_transformed = scaler.fit_transform(X_new)

    # 4. PCA si configurée
    pca_config = pipeline.get("pca")
    pca_n_components = pipeline.get("pca_n_components")
    if pca_config is not None:
        # De même, charger votre PCA entraîné
        pca = PCA(n_components=pca_n_components)  # Remplacez par votre vrai PCA chargé
        X_transformed = pca.fit_transform(X_transformed)
    # 5. Correction batch si configurée
    combat_config = pipeline.get("combat")
    if combat_config:
        feature_names = [f"var_{i}" for i in range(X_transformed.shape[1])]
        data = pd.DataFrame(X_transformed, columns=feature_names)
        X_transformed = pycombat(data.T, batch).T
        print(f"---->>> Correction batch appliquée (ComBat)")
    return X_transformed