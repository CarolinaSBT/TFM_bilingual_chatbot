# TFM_bilingual_chatbot
Repositorio de código para mi TFM (un chatbot que responde en inglés y español).
## Chatbot App API
Este repositorio contiene el código de una API construida con Flask que sirve como backend para el chatbot. La API se encarga de manejar las inferencias de los modelos de clasificación de intenciones y de reconocimiento de entidades nombradas (NER).

🚀 **Características principales**
- Clasificación de intenciones: identifica la intención del usuario a partir del texto de entrada.
- Reconocimiento de entidades nombradas (NER): extrae entidades (ej. productos, categorías) del texto.
- Integración: diseñado para ser un servicio de backend ligero que interactúa con el frontend del chatbot.

📂 **Estructura del repositorio**
- app.py: el archivo principal de la aplicación Flask.
- requirements.txt: lista de dependencias de Python.
- dockerfile: permite realizar el despliegue en un contenedor Docker
- models/: directorio donde se almacenan los modelos.
- data/: contiene una muestra del dataset de intents original desde donde se extraen los labels
