from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

# <<<<<<<<<<<<<<<< IMPORTACIÓN CRÍTICA CORREGIDA >>>>>>>>>>>>>>>>>>
# Importamos la lista de CLASES y el tamaño de imagen (IMG_SIZE) desde el archivo de configuración,
# asegurando que la API se sincronice con el modelo entrenado (120 razas).
from config.config import CLASSES, IMG_SIZE 
# <<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>

# Diccionario global para gestionar recursos
ml_models = {}

# --- Construcción de Ruta Robusta al Modelo ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.keras')

# --- Lifespan (Gestión de Ciclo de Vida) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Al iniciar la API
    if os.path.exists(MODEL_PATH):
        print(f"✅ [Inicio] Cargando modelo desde: {MODEL_PATH}")
        try:
            ml_models["model"] = tf.keras.models.load_model(MODEL_PATH)
            print("🚀 [Inicio] Modelo cargado en memoria y listo.")
        except Exception as e:
            print(f"❌ [Error] Fallo al cargar el modelo: {e}")
    else:
        print(f"⚠️ [Aviso] No se encontró {MODEL_PATH}. La API iniciará pero fallará al predecir.")
    
    yield # Aquí corre la aplicación
    
    # 2. Al apagar la API
    ml_models.clear()
    print("🛑 [Apagado] Memoria liberada.")

app = FastAPI(title="AnimalNet API", description="API de Clasificación Profesional", lifespan=lifespan)


def prepare_image(image_bytes):
    """Preprocesa la imagen binaria para EfficientNet"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Usamos IMG_SIZE importado de config para asegurar la dimensión correcta (224, 224)
    img = img.resize(IMG_SIZE) 
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Batch de 1
    img_array = preprocess_input(img_array)
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validación de "Inferencia Caliente"
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Servicio no disponible.")
    
    try:
        image_bytes = await file.read()
        processed_image = prepare_image(image_bytes)
        
        # Inferencia
        prediction = ml_models["model"].predict(processed_image)
        predicted_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # El acceso a CLASSES[predicted_idx] ahora es seguro (120 nombres)
        return {
            "filename": file.filename,
            "prediction": CLASSES[predicted_idx],
            "confidence": round(confidence, 4),
            "probabilities": {k: float(v) for k, v in zip(CLASSES, prediction[0])}
        }
    except Exception as e:
        # El error 'list index out of range' queda resuelto, si hay otro error, se reporta.
        raise HTTPException(status_code=500, detail=f"Error interno procesando imagen: {str(e)}")

@app.get("/")
def root():
    return {"status": "online", "model_loaded": "model" in ml_models}