import os
from flask import Flask, request, jsonify

# Importa las librerías necesarias
import torch
import pickle
# Mantener esta línea por si se necesita para el tipo de objeto
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Crea la aplicación Flask
app = Flask(__name__)

# --- Carga del modelo ---
# Se carga el modelo y el tokenizador de forma global, una sola vez al inicio.
try:
    print("Intentando cargar el modelo y el tokenizador desde un solo archivo .pkl...")

    # Define la ruta a tu archivo .pkl en el directorio local de Render
    # Asume que el archivo contiene una tupla o un diccionario con ambos objetos
    model_tokenizer_path = "models/intent_classification_model_v1/intent_classification_model.pkl"

    # Usa 'rb' para leer en modo binario
    with open(model_tokenizer_path, 'rb') as file:
        model, tokenizer = pickle.load(file)

    print("¡Modelo y tokenizador cargados con éxito!")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo del modelo. Por favor, revisa la ruta. {e}")
    model = None
    tokenizer = None
except Exception as e:
    print(f"Error al cargar el archivo .pkl: {e}")
    model = None
    tokenizer = None


# --- Funciones de la API ---

@app.route('/', methods=['GET'])
def welcome():
    """
    Maneja la petición a la URL raíz y da la bienvenida.
    """
    return "¡Bienvenido a la API del chatbot! La API para clasificar texto está en la ruta /predict"


@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Función principal para la predicción de intents.
    Recibe un JSON con 'text_input' y devuelve el intent clasificado.
    """
    # Verifica si el modelo se cargó correctamente
    if model is None or tokenizer is None:
        return jsonify({
            "error": "El modelo no se pudo cargar. Por favor, revisa los logs del servidor."
        }), 500

    try:
        # Obtiene el cuerpo de la petición en formato JSON
        data = request.get_json()

        # Extrae el texto del campo 'text_input'
        text_input = data.get('text_input', '')

        # Codifica el texto de entrada con el tokenizador cargado
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)

        # Realiza la inferencia del modelo
        with torch.no_grad():
            outputs = model(**inputs)

        # Obtiene las probabilidades y la etiqueta predicha
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1).item()

        # Mapea el ID de la etiqueta a su nombre de intent
        label_map = {
            0: "animal_funfact", 1: "bye", 2: "cancel_task",
            3: "confirm", 4: "greetings", 5: "human_help", 6: "negate",
            7: "other", 8: "question_delivery", 9: "question_item",
            10: "question_offers", 11: "question_prices", 12: "question_returns",
            13: "question_store", 14: "thank_you"
        }

        # Obtiene el nombre del intent a partir del mapa, con un valor por defecto
        classification_result = label_map.get(predicted_label, 'desconocido')

        # Devuelve la respuesta en formato JSON
        return jsonify({
            "data": [classification_result]
        })

    except Exception as e:
        # Maneja cualquier error durante el procesamiento y lo devuelve
        return jsonify({
            "error": f"Ocurrió un error durante el procesamiento: {e}"
        }), 500


if __name__ == '__main__':
    # Elige el puerto proporcionado por Render o por defecto 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
