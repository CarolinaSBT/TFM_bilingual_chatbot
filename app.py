import os
from flask import Flask, request, jsonify

app = Flask(__name__)


# Simula la carga de un modelo, pero no usa memoria
# Reemplaza esta función con la tuya una vez que confirmes que funciona
def run_classification(text_input):
    """
    Función de prueba que simula la clasificación.
    """
    return f"El texto recibido es: {text_input}"


@app.route('/predict', methods=['POST'])
def handle_prediction():
    # Obtiene el cuerpo de la petición en formato JSON
    data = request.get_json()

    # Extrae el texto del campo 'text_input'
    text_input = data.get('text_input', '')

    # Llama a la función de clasificación
    classification_result = run_classification(text_input)

    # Devuelve la respuesta en formato JSON
    return jsonify({
        "data": [classification_result]
    })


if __name__ == '__main__':
    # Elige el puerto proporcionado por Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

