#!/usr/bin/env bash

# Descarga el modelo de spaCy necesario
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm

# Inicia tu aplicación con Gunicorn
gunicorn app:app