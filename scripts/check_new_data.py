import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

def check_new_data(raw_data_dir):
    raw_dir = Path(raw_data_dir)
    new_files = [f for f in raw_dir.glob('*.csv') if f.stat().st_mtime > (datetime.now() - timedelta(days=100)).timestamp()]
    
    if not new_files:
        logging.info("Aucun nouveau fichier détecté")
        raise ValueError("Aucun nouveau fichier à traiter")
    
    logging.info(f"Nouveaux fichiers détectés: {new_files}")
    return str(new_files[0])