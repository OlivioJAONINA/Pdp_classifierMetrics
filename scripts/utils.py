import importlib
import yaml
from sklearn.model_selection import GridSearchCV


def load_config(config_path="scripts/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_estimator(name, class_path):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


def grid_search_model(estimator, param_grid, X_train, y_train):
    grid = GridSearchCV(estimator, param_grid, scoring='f1_weighted', cv=5)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_


from mlflow.tracking import MlflowClient
import mlflow
import logging

def delete_last_experiments(mlflow_uri, keep_last_n=2, delete_n_before_last=2):
    """
    Supprime les expériences juste avant les `n` dernières conservées (par ordre de création).

    Args:
        mlflow_uri (str): URI du serveur MLflow.
        keep_last_n (int): Nombre d'expériences les plus récentes à conserver.
        delete_n_before_last (int): Nombre d'expériences à supprimer juste avant les plus récentes.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    # Récupère toutes les expériences visibles (non supprimées)
    experiments = client.list_experiments(view_type=1)  # 1 = active only

    # Trie par experiment_id (souvent incrémental, approximatif pour date)
    experiments_sorted = sorted(experiments, key=lambda e: int(e.experiment_id))

    total = len(experiments_sorted)
    if total < keep_last_n + delete_n_before_last:
        logging.warning("Pas assez d'expériences pour en supprimer autant.")
        return

    to_delete = experiments_sorted[-(keep_last_n + delete_n_before_last):-keep_last_n]

    for exp in to_delete:
        logging.info(f"Suppression de l'expérience : {exp.name} (ID: {exp.experiment_id})")
        client.delete_experiment(exp.experiment_id)


import mlflow
from mlflow.tracking import MlflowClient

def keep_only_last_two_experiments(dry_run=True):
    """
    Garde seulement les 2 expériences MLflow les plus récentes et supprime toutes les autres.
    
    Args:
        dry_run (bool): Si True, affiche seulement ce qui serait supprimé sans effectuer les suppressions.
    
    Returns:
        dict: Statistiques des opérations effectuées
    """
    client = MlflowClient()
    stats = {
        'total_experiments': 0,
        'deleted_experiments': 0,
        'kept_experiments': 0,
        'kept_experiment_ids': [],
        'deleted_experiment_ids': []
    }
    
    try:
        # Récupérer toutes les expériences triées par date de création (plus récent d'abord)
        experiments = sorted(
            client.search_experiments(),
            key=lambda x: x.creation_time or 0,
            reverse=True
        )
        
        stats['total_experiments'] = len(experiments)
        
        if len(experiments) <= 2:
            print("Il y a 2 expériences ou moins - aucune suppression nécessaire.")
            stats['kept_experiments'] = len(experiments)
            stats['kept_experiment_ids'] = [exp.experiment_id for exp in experiments]
            return stats
        
        # Les deux à garder
        to_keep = experiments[:2]
        stats['kept_experiments'] = len(to_keep)
        stats['kept_experiment_ids'] = [exp.experiment_id for exp in to_keep]
        
        # Les autres à supprimer
        to_delete = experiments[2:]
        
        for exp in to_delete:
            if dry_run:
                print(f"[DRY RUN] Suppression de l'expérience: {exp.name} (ID: {exp.experiment_id})")
            else:
                try:
                    client.delete_experiment(exp.experiment_id)
                    print(f"Suppression de l'expérience: {exp.name} (ID: {exp.experiment_id})")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {exp.name}: {str(e)}")
                    continue
            
            stats['deleted_experiment_ids'].append(exp.experiment_id)
            stats['deleted_experiments'] += 1
        
    except Exception as e:
        print(f"Erreur lors du nettoyage des expériences: {str(e)}")
    
    return stats