import os
import json
import re
import torch
import spacy
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configura la aplicación de Flask
app = Flask(__name__)

# --- Rutas de los modelos ---
# Asegúrate de que estas rutas sean relativas al directorio raíz de tu proyecto
NER_ES_MODEL_PATH = "./models/ner_model_v1/output_es/model-best"
NER_EN_MODEL_PATH = "./models/ner_model_v1/output_eng/model-best"
INTENT_CLASSIFICATION_PATH = "./models/intent_classification_model_v1"
INTENTS_DB_PATH = "./data/Intents_db_v2.csv"

# --- Carga de modelos y preprocesamiento ---
# Carga el tokenizer de clasificación de intents
try:
    # `local_files_only=True` es crucial para que el modelo se cargue desde el
    # contenedor y no intente descargarlo de internet.
    loaded_tokenizer = AutoTokenizer.from_pretrained(INTENT_CLASSIFICATION_PATH, local_files_only=True)
    print("Tokenizer de clasificación de intents cargado.")
except Exception as e:
    print(f"Error al cargar el tokenizer: {e}")
    loaded_tokenizer = None

# Carga el modelo de clasificación de intents
try:
    loaded_model = AutoModelForSequenceClassification.from_pretrained(INTENT_CLASSIFICATION_PATH, local_files_only=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded_model.to(device)
    print(f"Modelo de clasificación de intents cargado y movido a dispositivo: {device}")
except Exception as e:
    print(f"Error al cargar el modelo de clasificación de intents: {e}")
    loaded_model = None
    device = torch.device('cpu') # Asegura que `device` esté definido

# Carga el archivo de intents e inicializa el LabelEncoder
df = None
label_encoder = None
try:
    df = pd.read_csv(INTENTS_DB_PATH, delimiter=';')
    label_encoder = LabelEncoder()
    df['encoded_intent'] = label_encoder.fit_transform(df['intent'])
    print("Base de datos de intents cargada y LabelEncoder inicializado.")
except Exception as e:
    print(f"Error al cargar la base de datos de intents o inicializar LabelEncoder: {e}")
    df = None
    label_encoder = None

# Carga los modelos de spaCy (NER)
ner_extractor_en = None
ner_extractor_es = None
try:
    ner_extractor_es = spacy.load(NER_ES_MODEL_PATH)
    print(f"Modelo español de spaCy cargado correctamente desde {NER_ES_MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo español de spaCy: {e}")

try:
    ner_extractor_en = spacy.load(NER_EN_MODEL_PATH)
    print(f"Modelo inglés de spaCy cargado correctamente desde {NER_EN_MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo inglés de spaCy: {e}")


# --- Funciones de procesamiento ---

def preprocess_text(text):
    """
    Preprocesa el texto de entrada:
    - Elimina signos de puntuación iniciales.
    - Convierte el texto a minúsculas.
    """
    processed_text = re.sub(r'^[¿¡\.$]', '', text)
    processed_text = processed_text.lower()
    return processed_text

#  --- Función para predecir intents con umbral de confianza ---
confidence_threshold = 0.6

def predict_with_confidence_threshold(text):
    """
    Predice el intent de un texto dado con un umbral de confianza.
    """
    # Verifica que todos los modelos necesarios estén cargados
    if loaded_model is None or loaded_tokenizer is None or label_encoder is None:
        return {"error": "Modelos de clasificación de intents no cargados."}

    # Preprocesamiento del texto
    processed_text = preprocess_text(text)

    # Tokenización y movimiento a `device`
    inputs = loaded_tokenizer(
        processed_text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Realiza la predicción
    with torch.no_grad():
        outputs = loaded_model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    max_prob, predicted_id = torch.max(probabilities, dim=-1)

    max_prob_value = max_prob.item()
    predicted_id_value = predicted_id.item()
    predicted_intent_name = label_encoder.inverse_transform([predicted_id_value])[0]

    # Decide si el intent predicho es "other" o el real basado en la confianza
    final_intent = predicted_intent_name if max_prob_value >= confidence_threshold else "other"

    all_probabilities = {label: prob.item() for label, prob in zip(label_encoder.classes_, probabilities)}

    return {
        "text": text,
        "intent": final_intent,
        "confidence": max_prob_value,
        "all_probabilities": all_probabilities
    }

# --- Endpoints de la API ---
@app.route("/", methods=["GET"])
def health_check():
    return "API de NER e Intención está activa. Usa el endpoint /predict para hacer una solicitud POST."

@app.route("/predict", methods=["POST"])
def predict():
    # Verifica que todos los modelos necesarios para la API estén cargados al inicio
    if loaded_model is None or ner_extractor_es is None or ner_extractor_en is None:
        return jsonify({"error": "Uno o más modelos no se cargaron correctamente. Por favor, revise los logs de implementación."}), 500

    try:
        # Obtiene los datos de la solicitud
        data = request.get_json()
        user_message = data.get("message")
        lang = data.get("lang", "spanish") # Valor por defecto 'spanish'

        if not user_message:
            return jsonify({"error": "El campo 'message' es obligatorio en el cuerpo de la solicitud."}), 400

        # Paso 1: Predicción de intenciones
        intent_prediction = predict_with_confidence_threshold(user_message)

        if "error" in intent_prediction:
            return jsonify(intent_prediction), 500

        # Paso 2: Extracción de entidades (NER)
        entities = []
        if lang.lower() == "spanish":
            doc = ner_extractor_es(user_message)
            for ent in doc.ents:
                entities.append({
                    "entidad": ent.text,
                    "es": ent.label_
                })
        elif lang.lower() == "english":
            doc = ner_extractor_en(user_message)
            for ent in doc.ents:
                entities.append({
                    "entity": ent.text,
                    "is": ent.label_
                })
        else:
            return jsonify({"error": f"Idioma no soportado: '{lang}'. Idiomas válidos: 'spanish' o 'english'."}), 400

        # Paso 3: Combinar resultados y devolver la respuesta
        response_data = {
            "intent": intent_prediction["intent"],
            "confidence": intent_prediction["confidence"],
            "entities": entities
        }
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error inesperado al procesar la solicitud: {e}")
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500

# Punto de entrada principal para Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
