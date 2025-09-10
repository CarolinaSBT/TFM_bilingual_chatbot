# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias necesarias para el build, como el compilador de C
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de requerimientos y los instala
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Establece una variable de entorno para el puerto
# El valor de 8080 es el puerto por defecto de Cloud Run
ENV PORT=8080

# Comando para ejecutar la aplicación cuando el contenedor se inicie
# Gunicorn para manejar las peticiones HTTP
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app