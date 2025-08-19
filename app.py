import os
from flask import Flask, request, jsonify

# Importa spaCy
import spacy

# Crea la aplicación Flask
app = Flask(__name__)

# --- Carga del modelo ---
# Se carga el modelo de spaCy de forma global, una sola vez al inicio.
try:
    print("Intentando cargar el modelo NER de spaCy desde un directorio...")

    # Define la ruta a tu directorio de modelo
    # ¡Asegúrate de que esta ruta sea correcta para tu proyecto!
    model_path = "models/ner_model_v1/output_es/model-best"

    # Carga el modelo desde el directorio
    nlp = spacy.load(model_path)

    print("¡Modelo de spaCy cargado con éxito!")

except Exception as e:
    print(f"Error al cargar el modelo de spaCy: {e}")
    nlp = None


# --- Funciones de la API ---

@app.route('/', methods=['GET'])
def welcome():
    """
    Maneja la petición a la URL raíz y da la bienvenida.
    """
    return "¡Bienvenido a la API de NER! La API para extraer entidades está en la ruta /predict"


@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Función principal para la predicción de entidades con spaCy.
    Recibe un JSON con 'text_input' y devuelve las entidades reconocidas.
    """
    # Verifica si el modelo se cargó correctamente
    if nlp is None:
        return jsonify({
            "error": "El modelo de spaCy no se pudo cargar. Por favor, revisa los logs del servidor."
        }), 500

    try:
        # Obtiene el cuerpo de la petición en formato JSON
        data = request.get_json()

        # Extrae el texto del campo 'text_input'
        text_input = data.get('text_input', '')

        # Procesa el texto con el modelo de spaCy
        doc = nlp(text_input)

        # Extrae las entidades y las formatea
        entities = []
        for ent in doc.ents:
            entities.append({
                'entity_text': ent.text,
                'entity_type': ent.label_
            })

        # Devuelve la respuesta en formato JSON
        return jsonify({
            "data": entities
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
