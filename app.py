from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle  # Importamos la librería para cargar el modelo

# Carga tu modelo de forma global para que se cargue solo una vez
try:
    print("Intentando cargar el modelo...")
    # La ruta al archivo .pkl en tu repositorio.
    model_path = "models/intent_classification_model_v1/intent_classification_model.pkl"

    with open(model_path, 'rb') as f:
        # Asumimos que el archivo .pkl contiene un diccionario
        # con las claves 'model' y 'tokenizer'
        loaded_data = pickle.load(f)
        model = loaded_data['model']
        tokenizer = loaded_data['tokenizer']

    print("¡Modelo y tokenizador cargados con éxito!")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None
    tokenizer = None


# Define la función que procesará las entradas y dará la salida
def run_classification_logic(text_input):
    """
    Función para realizar la clasificación de texto.
    """
    if model is None or tokenizer is None:
        return "El modelo no se pudo cargar. Por favor, revisa los logs de tu Space."

    try:
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1).item()

        label_map = {0: "animal_funfact", 1: "bye", 2: "cancel_task",
                     3: "confirm", 4: "greetings", 5: "human_help", 6: "negate",
                     7: "other", 8: "question_delivery", 9: "question_item",
                     10: "question_offers", 11: "question_prices", 12: "question_returns",
                     13: "question_store", 14: "thank_you"}

        return label_map.get(predicted_label, 'desconocido')

    except Exception as e:
        return f"Ocurrió un error durante el procesamiento: {e}"


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def handle_prediction_request():
    # Obtiene el cuerpo de la petición en formato JSON
    data = request.get_json()

    # Extrae el texto del campo 'text_input'
    text_input = data.get('text_input', '')

    # Llama a la función de clasificación para obtener el resultado real
    classification_result = run_classification_logic(text_input)

    # Devuelve la respuesta en formato JSON
    return jsonify({
        "data": [classification_result]
    })


if __name__ == '__main__':
    app.run(debug=True)
