# Image de base ultra-légère
FROM python:3.8-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR 1

# Installation des dépendances minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Installation de MLflow avec le strict minimum
RUN pip install --no-cache-dir \
    mlflow==2.3.2 \
    psycopg2-binary==2.9.6 \
    sqlalchemy==1.4.46 \
    scikit-learn==1.2.2 \
    xgboost==1.7.5 \
    lightgbm==3.3.2 

# Port exposé
EXPOSE 5001

# Point d'entrée
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5001"]
