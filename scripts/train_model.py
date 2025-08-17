import logging
from datetime import datetime
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from scripts.utils import (
    load_config,
    get_estimator,
    grid_search_model,
    keep_only_last_two_experiments,
)
from mlflow.exceptions import MlflowException
import numpy as np


def f2_score_func(y_true, y_pred):
    """
    Calcule le F2-score pour la classification binaire.
    """
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    if (4 * f1 + recall) == 0:
        return 0.0
    return (5 * f1 * recall) / (4 * f1 + recall)


def specificity_score(y_true, y_pred):
    """
    Calcule la spécificité pour la classification binaire.
    """
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2:
        tn, fp, _, _ = cm.ravel()
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)
    return 0.0


def train_and_compare(
    data_path,
    models_dir,
    mlflow_uri,
    target_var,
    experiment_name,
    config_path,
    export_dir=None
):
    # Configuration de MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    keep_only_last_two_experiments(dry_run=True)
    mlflow.set_experiment(experiment_name + str(datetime.now().strftime("%Y%m%d_%H%M%S")))

    # Chargement des données
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-5].values
    y = df[target_var].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chargement de la config
    config = load_config(config_path)

    # Préparation des dossiers
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model.pkl"

    if export_dir:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        exported_model_path = export_dir / "best_model.pkl"
    else:
        exported_model_path = None

    best_model = None
    best_scores = None
    best_model_name = ""

    # Déterminer si c'est une classification binaire ou multiclasse
    is_binary = (len(np.unique(y)) == 2) and (target_var.lower() == "sick")

    # Entraînement et évaluation des modèles
    for name, model_conf in config["models"].items():
        # S'assurer que le paramètre probability est activé si nécessaire
        if name.lower() == "svc" and 'probability' not in model_conf['estimator']:
            model_conf['estimator']['probability'] = True
        
        # Corriger les types de paramètres si nécessaire
        for param, value in model_conf["param_grid"].items():
            if isinstance(value, list):
                model_conf["param_grid"][param] = [
                    int(v) if isinstance(v, str) and v.isdigit() else v for v in value
                ]

        estimator = get_estimator(name, model_conf["estimator"])
        with mlflow.start_run(run_name=name):
            model, cv_score = grid_search_model(estimator, model_conf["param_grid"], X_train, y_train)
            y_pred = model.predict(X_test)

            if is_binary: # Pour 'sick'
                f2_score_val = f2_score_func(y_test, y_pred)
                f1_score_val = f1_score(y_test, y_pred, average="weighted")
                recall_val = recall_score(y_test, y_pred, average="weighted")
                specificity_val = specificity_score(y_test, y_pred)
                
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc_roc_val = roc_auc_score(y_test, y_proba)
                except (AttributeError, ValueError):
                    logging.warning(f"AUC-ROC non calculée pour {name}. predict_proba non disponible.")
                    auc_roc_val = 0.0

                current_scores = (f2_score_val, recall_val, specificity_val)

                # Log des métriques dans l'ordre spécifié
                mlflow.log_metric("F2-score", f2_score_val)
                mlflow.log_metric("F1-score", f1_score_val)
                mlflow.log_metric("Recall", recall_val)
                mlflow.log_metric("Specificity", specificity_val)
                mlflow.log_metric("AUC-ROC", auc_roc_val)
                mlflow.log_metric("cv_f1_score", cv_score)
            else: # Pour 'pathologic'
                f1_macro_val = f1_score(y_test, y_pred, average="macro")
                f1_score_val = f1_score(y_test, y_pred, average="weighted")
                recall_macro_val = recall_score(y_test, y_pred, average="macro")
                recall_val = recall_score(y_test, y_pred, average="weighted")
                specificity_val = specificity_score(y_test, y_pred)
                
                try:
                    y_proba = model.predict_proba(X_test)
                    auc_roc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except (AttributeError, ValueError):
                    logging.warning(f"AUC-ROC non calculée pour {name}. predict_proba non disponible.")
                    auc_roc_val = 0.0

                current_scores = (f1_macro_val, recall_macro_val)
                
                # Log des métriques dans l'ordre spécifié
                mlflow.log_metric("F1-Macro", f1_macro_val)
                mlflow.log_metric("F1-score", f1_score_val)
                mlflow.log_metric("Recall-Macro", recall_macro_val)
                mlflow.log_metric("Recall", recall_val)
                mlflow.log_metric("Specificity", specificity_val)
                mlflow.log_metric("AUC-ROC", auc_roc_val)
                mlflow.log_metric("cv_f1_score", cv_score)


            mlflow.sklearn.log_model(model, "model")

            # Comparaison lexicographique (inchangée)
            if best_scores is None or current_scores > best_scores:
                best_scores = current_scores
                best_model = model
                best_model_name = name

    # Comparaison avec l'ancien modèle
    old_scores = None
    if best_model_path.exists():
        try:
            old_model = joblib.load(best_model_path)
            old_pred = old_model.predict(X_test)
            
            if is_binary:
                old_f2 = f2_score_func(y_test, old_pred)
                old_recall = recall_score(y_test, old_pred, average="weighted")
                old_specificity = specificity_score(y_test, old_pred)
                
                try:
                    old_proba = old_model.predict_proba(X_test)[:, 1]
                    old_auc_roc = roc_auc_score(y_test, old_proba)
                except (AttributeError, ValueError):
                    old_auc_roc = 0.0

                old_scores = (old_f2, old_recall, old_specificity, old_auc_roc)
            else:
                old_f1_macro = f1_score(y_test, old_pred, average="macro")
                old_recall_macro = recall_score(y_test, old_pred, average="macro")
                old_scores = (old_f1_macro, old_recall_macro)
                
        except Exception as e:
            logging.warning(f"Ancien modèle non chargé : {e}")

    # Décision finale
    with mlflow.start_run(run_name="Final_Model_Selection"):
        mlflow.log_param("selected_model", best_model_name)
        if best_scores:
            if is_binary:
                # Log des métriques dans l'ordre spécifié
                mlflow.log_metric("new_F2-score", best_scores[0])
                mlflow.log_metric("new_F1-score", f1_score(y_test, best_model.predict(X_test), average="weighted"))
                mlflow.log_metric("new_Recall", best_scores[1])
                mlflow.log_metric("new_Specificity", best_scores[2])
                try:
                    new_auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
                    mlflow.log_metric("new_AUC-ROC", new_auc_roc)
                except (AttributeError, ValueError):
                    mlflow.log_metric("new_AUC-ROC", 0.0)
            else:
                # Log des métriques dans l'ordre spécifié
                mlflow.log_metric("new_F1-Macro", best_scores[0])
                mlflow.log_metric("new_F1-score", f1_score(y_test, best_model.predict(X_test), average="weighted"))
                mlflow.log_metric("new_Recall-Macro", best_scores[1])
                mlflow.log_metric("new_Recall", recall_score(y_test, best_model.predict(X_test), average="weighted"))
                mlflow.log_metric("new_Specificity", specificity_score(y_test, best_model.predict(X_test)))
                try:
                    new_auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr', average='weighted')
                    mlflow.log_metric("new_AUC-ROC", new_auc_roc)
                except (AttributeError, ValueError):
                    mlflow.log_metric("new_AUC-ROC", 0.0)

        gain_made = False
        if best_scores is not None:
            if old_scores is None:
                gain_made = True
            elif is_binary:
                if best_scores[0] > old_scores[0] or (best_scores[0] == old_scores[0] and best_scores[1] > old_scores[1]) or (best_scores[0] == old_scores[0] and best_scores[1] == old_scores[1] and best_scores[2] > old_scores[2]):
                    gain_made = True
            else:
                if best_scores[0] > old_scores[0] or (best_scores[0] == old_scores[0] and best_scores[1] > old_scores[1]):
                    gain_made = True

        if gain_made:
            joblib.dump(best_model, best_model_path)
            logging.info(f"✅ Nouveau meilleur modèle sauvegardé dans {best_model_path} ({best_model_name})")

            if exported_model_path:
                joblib.dump(best_model, exported_model_path)
                logging.info(f"📁 Modèle également exporté dans {exported_model_path}")

            return str(best_model_path)
        else:
            logging.info("⏸️ Aucun gain. Ancien modèle conservé.")
            return None