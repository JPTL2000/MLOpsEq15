# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia todo lo necesario al contenedor
COPY src/api/predict.py src/api/
COPY src/api/model_extraction.py src/api/
COPY src/model_name.txt src/
COPY src/model_metadata.txt src/
COPY models/ models/

# (Opcional) Instala dependencias si tu modelo las necesita
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto al iniciar el contenedor
WORKDIR /app/src/api
#CMD ["python", "predict.py"]
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
