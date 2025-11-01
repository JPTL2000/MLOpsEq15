# Descripción del proyecto
Este proyecto de Data Science / MLOps tiene como objetivo predecir si un artículo en línea se volverá viral o no, utilizando el dataset Online News Popularity del UCI Machine Learning Repository. El modelo busca apoyar estrategias publicitarias y de contenido mediante la identificación temprana de publicaciones con alto potencial de viralización.

# Objetivos
- Analizar las características de los artículos que influyen en su popularidad.
- Construir un modelo de clasificación binaria (viral / no viral).
- Implementar un flujo reproducible de Data Science con prácticas de MLOps.

# Estructura del proyecto
El repositorio sigue la estructura de CookieCutter Data Science, la cual facilita la organización modular y escalable del flujo de trabajo.

# Setup del entorno
### Setup de entorno virtual de Python:
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
pip install -r requirements.txt
(Solo usar el segundo comando si no funciona el tercero)

### Setup de DVC
dvc remote add -f myremote s3://dvc-storage-mlops-eq15
dvc remote modify myremote region us-east-2
dvc remote modify myremote access_key_id agregar
dvc remote modify myremote secret_access_key agregar
dvc pull -r myremote

### Uso de DVC
dvc pull -r myremote
dvc add folder-con-nuevos-archivos
dvc push -r myremote

# Herramientas y tecnologías
- Python
- DVC (Data Version Control) para versionar datasets y modelos.
- Amazon S3 como almacenamiento remoto de datos y artefactos.
- CookieCutter Data Science como plantilla de estructura de proyecto.
- MLFlow para el registro de experimentos.

# Flujo MLOps
El ciclo de vida del proyecto se gestiona mediante DVC, garantizando trazabilidad y reproducibilidad:
1. Datos crudos → se almacenan y versionan con DVC.
2. Procesamiento y feature engineering → scripts reproducibles en /src.
3. Entrenamiento del modelo → resultados almacenados localmente y en S3.
4. Versionado de modelos y métricas → mediante commits y dvc push.
5. Integración con MLFlow para seguimiento de experimentos.

# Resultados esperados
- Clasificador capaz de identificar artículos con alto potencial de viralización.
- Reporte de importancia de variables y patrones de comportamiento.
- Pipeline reproducible y automatizable.