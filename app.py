import re
import os
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import spacy
from flask import Flask, request, jsonify
from supabase import create_client, Client
import io  # Import the io module
import json  # Import json to handle JSONB data

# Paths a los directorios y archivos relevantes
loaded_model_dir = 'models/intent_classification_model_v1'
path_intents = 'data/Intents_db_v2.csv'
ner_model_en_path = 'models/ner_model_v1/output_eng/model-best'
ner_model_es_path = 'models/ner_model_v1/output_es/model-best'

# --- Configuración Supabase ---
# Acceso a claves de Supabase guardadas como claves de entorno en HF
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

# Inicializa Supabase si las claves están disponibles
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client inicializado.")
    except Exception as e:
        print(f"Error la inicializar Supabase client: {e}")
        # Continúa sin Supabase si falla la inicialización

# Nombre de la tabla en Supabase
SUPABASE_LOG_TABLE = "prediction_logs"


# Cargar el tokenizer
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(loaded_model_dir, local_files_only=True)
    print("Tokenizer cargado.")
except Exception as e:
    print(f"Error al cargar tokenizer: {e}")
    loaded_tokenizer = None

# Cargar el modelo de clasificación de intents
try:
    loaded_model = DistilBertForSequenceClassification.from_pretrained(loaded_model_dir, local_files_only=True)
    print("Model cargado.")
except Exception as e:
    print(f"Error al cargar modelo: {e}")
    loaded_model = None

# Si el modelo está cargado y hay un gpu disponible, lo mueve
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if loaded_model:
    loaded_model.to(device)
    print(f"Model moved to device: {device}")


# Carga el archivo de intents e inicializa el label encoder
df = None
label_encoder = None
try:
    df = pd.read_csv(path_intents, delimiter=';')
    label_encoder = LabelEncoder()
    df['encoded_intent'] = label_encoder.fit_transform(df['intent'])
    print("Intents y label cargados.")
except Exception as e:
    print(f"Error al cargar intents o label encoder: {e}")
    df = None
    label_encoder = None

# Carga los modelos NER
ner_extractor_en = None
ner_extractor_es = None
try:
    ner_extractor_en = spacy.load(ner_model_en_path)
    print(f"Se cargó el modelo NER en inglés de {ner_model_en_path}")
except Exception as e:
    print(f"Error al cargar modelo NER en inglés {ner_model_en_path}: {e}")
    ner_extractor_en = None

try:
    ner_extractor_es = spacy.load(ner_model_es_path)
    print(f"Se cargó el modelo NER en español de {ner_model_es_path}: {e}")
except Exception as e:
    print(f"Error al cargar modelo NER en español {ner_model_es_path}: {e}")
    ner_extractor_es = None


# Función para preprocesar texto
def preprocess_text(text):
    """
    Preprocesa el texto de entrada:
    - Elimina signos de admiración y exclamación al inicio de las oraciones.
    - Elimina puntos finales al final de las oraciones.
    - Convierte el texto a minúsculas.

    Args:
      text (str): El texto de entrada para preprocesar.

    Returns:
      str: El texto preprocesado.
    """
    processed_text = text
    # Eliminar signos de admiración y exclamación al inicio
    processed_text = re.sub(r'[¿¡\.$]', '', processed_text)
    # Convertir a minúsculas
    processed_text = processed_text.lower()
    return processed_text


# Confidence threshold
confidence_threshold = 0.6


# Función para predecir intents
def predict_with_confidence_threshold(text, model, tokenizer, label_encoder, device, confidence_threshold=confidence_threshold, max_len=128):
    """ Predice el intent de un texto dado con un umbral de confianza.
        Utiliza la función preprocess_text para limpiar el texto antes de la predicción.

    Args:
      text (str): El texto de entrada para predecir el intent.

    Returns: texto de entrada, predicción del intent con mayor confianza,
           grados de confianza de todas las categorías.
           Devuelve None si no se pudo cargar el modelo o el tokenizer.
    """
    if model is None or tokenizer is None or label_encoder is None:
        print("No se cargaron los modelos necesarios. No es posible predecir.")
        return None

    # Preprocesamiento del texto
    processed_text = preprocess_text(text)

    inputs = tokenizer(processed_text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Realiza la predicción
    with torch.no_grad():  # Deshabilita la actualización de gradientes
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    max_prob, predicted_id = torch.max(probabilities, dim=-1)

    max_prob_value = max_prob.item()  # Define el intent con más probabilidades
    predicted_id_value = predicted_id.item()
    predicted_intent_name = label_encoder.inverse_transform([predicted_id_value])[0]  # Recupera los labels de los intents

    if max_prob_value >= confidence_threshold:
        final_intent = predicted_intent_name
    else:
        final_intent = "other"

    all_probabilities = {label: prob.item() for label, prob in zip(label_encoder.classes_, probabilities)}

    return {
        "text": text, # Devolver el texto original para el registro
        "intent": final_intent,
        "confidence": max_prob_value,
        "all_probabilities": all_probabilities
    }


def log_prediction_to_supabase(prediction, entidades=None, lang=None, supabase_client: Client = None, table_name: str = None):
    """
    Guarda los resultados de las predicciones y entidades en una tabla de Supabase.

    Args:
        prediction (dict): diccionario que contiene las predicciones.
                           El diccionario debe tener las siguientes claves:
                           'text', 'intent', 'confidence', 'all_probabilities'.
        entidades (list): Lista de diccionarios con las entidades encontradas por NER.
        lang (str, optional): El idioma de la entrada.
        supabase_client (Client, optional): Cliente de Supabase inicializado.
        table_name (str, optional): Nombre de la tabla de Supabase para logs.

    Returns:
        dict: La respuesta de la inserción de Supabase, o None si hay un error o el cliente/tabla no están disponibles.
    """
    if prediction is None:
        print("No se recibió una predicción válida. No se puede guardar nada en Supabase.")
        return None

    if supabase_client is None or table_name is None:
        print("No se puede conectar con Supabase. No se inicializó la tabla o el client")
        return None

    try:
        # Preparar los datos
        log_data = {
            "user_message": prediction.get("text"),
            "predicted_intent": prediction.get("intent"),
            "confidence": prediction.get("confidence"),
            "all_probabilities": json.dumps(prediction.get("all_probabilities")),  # Convertir diccionario a json
            "entities": json.dumps(entidades) if entidades is not None else json.dumps([]),  # Convertir lista a json
            "language": lang  # Incluir idioma
        }

        # Insertar datos en la tabla de Supabase
        response = supabase_client.table(table_name).insert(log_data).execute()

        # Verificar si existen errores
        if hasattr(response, 'data') and response.data:
            print(f"Se añadió el registro a Supabase '{table_name}'.")
        else:
            print(f"No se insertaron logs. Revisar la tabla de Supabase'{table_name}'. Response: {response}")

        return response

    except Exception as e:
        print(f"Error al actualizar tabla '{table_name}': {e}")
        return None


app = Flask(__name__)


@app.route("/classify_intent_ner", methods=["POST"])
def classify_intent_api():
    # Revisar si se cargaron los modelos
    if loaded_tokenizer is None or loaded_model is None or label_encoder is None or (ner_extractor_en is None and ner_extractor_es is None):
        error_message = "No se cargaron los modelos. La API no está preparada."
        print(f"Error: {error_message}")
        return jsonify({"error": error_message}), 500

    try:
        # Recibe el mensaje del usuario en formato json
        data = request.get_json()
        user_message = data.get("message")
        lang = data.get("lang", "English") # Pone por defecto inglés, si no existe.

        if not user_message:
            print("Error: No llegó ingún mensaje")
            return jsonify({"error": "No llegó ningún mensaje."}), 400

        # Realizar la predicción
        # Pasar los componentes cargados a la función de predicción
        prediction = predict_with_confidence_threshold(
            user_message,
            loaded_model,
            loaded_tokenizer,
            label_encoder,
            device,
            confidence_threshold=confidence_threshold
        )

        # Comprobar si la predicción se generó
        if prediction is None:
            error_message = "La predicción falló."
            print(f"Error: {error_message}")
            return jsonify({"error": error_message}), 500

        #  NER
        entidades = []  # Inicializar la lista de entidades
        if lang.lower() == "spanish" and ner_extractor_es:
            doc = ner_extractor_es(user_message)
            for ent in doc.ents:
                entidades.append({
                    "Categoría": ent.label_,
                    "Palabra clave": ent.text
                })
        elif lang.lower() == "english" and ner_extractor_en:
            doc = ner_extractor_en(user_message) # Usar el modelo de inglés para inglés
            for ent in doc.ents:
                entidades.append({
                    "Category": ent.label_,
                    "Keyword": ent.text,
                })
        else:
            print(f"El modelo NER no está disponible para el idioma '{lang}'.")

        #  Cargar las predicciones en Supabase
        #  Pasar la tabla y el client name de Supabase
        log_prediction_to_supabase(
            prediction,
            entidades=entidades,
            lang=lang,  # Pasar idioma
            supabase_client=supabase,
            table_name=SUPABASE_LOG_TABLE
        )

        # Devolver la predicción y entidades en formato JSON
        print(f"Predicción para el mensaje: '{user_message}'")
        # Combina la predicción y las entidades en un solo json
        response_data = {
            "prediction": prediction,
            "entities": entidades
        }
        return jsonify(response_data), 200  # Devuelve 200 OK si hay success

    except Exception as e:
        print(f"Error in classify_intent_api: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask app...")

    # For Colab testing:
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error running Flask app locally: {e}")