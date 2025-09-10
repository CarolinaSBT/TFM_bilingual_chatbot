# TFM_bilingual_chatbot
Repositorio de código para mi TFM (un chatbot que responde en inglés y español).
## Chatbot App API
Este repositorio contiene el código de una API construida con Flask que sirve como backend para el chatbot. La API se encarga de manejar las inferencias de los modelos de clasificación de intenciones y de reconocimiento de entidades nombradas (NER).

🚀 **Características principales**
- Clasificación de intenciones: Identifica la intención del usuario a partir del texto de entrada.
- Reconocimiento de entidades nombradas (NER): Extrae entidades (ej. productos, categorías) del texto.
- Integración: Diseñado para ser un servicio de backend ligero que interactúa con el frontend del chatbot.

🛠️ **Requisitos de instalación**
Antes de correr la aplicación, es necesario tener instalados los siguientes componentes:
- Python 3.8+
- pip

**Instalación de dependencias**
1. Clonar este repositorio:
git clone [https://github.com/CarolinaSBT/TFM_bilingual_chatbot.git](https://github.com/CarolinaSBT/TFM_bilingual_chatbot.git)
cd TFM_bilingual_chatbot

3. Instalar las dependencias de Python:
pip install -r requirements.txt

📂 **Estructura del repositorio**
- app.py: El archivo principal de la aplicación Flask.
- requirements.txt: Lista de dependencias de Python.
- dockerfile: permite realizar el despliegue en un contenedor Docker
- models/: Directorio donde se almacenan los modelos.
- data/: contiene una muestra del dataset de intents original desde donde se extraen los labels
