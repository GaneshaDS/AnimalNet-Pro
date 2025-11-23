import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
# Asegúrate de que tu config.py tenga estas variables (debería tenerlas)
from config.config import MODEL_SAVE_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE

def evaluate():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ No se encontró el modelo en {MODEL_SAVE_PATH}. Entrena primero.")
        return

    # 1. Cargar modelo .keras
    print(f"📂 Cargando modelo desde {MODEL_SAVE_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return
    
    # 2. Crear Generador de Evaluación (Usando datos de VALIDACIÓN)
    # Usamos TRAIN_DIR pero extraemos el subset de validación para probar
    print("⚙️ Configurando generador de validación para pruebas...")
    
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2 # Debe coincidir con lo usado en el entrenamiento
    )

    # Importante: shuffle=False para que las etiquetas coincidan con las predicciones
    test_gen = eval_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation', 
        shuffle=False 
    )

    if test_gen.samples == 0:
        print("❌ Error: No se encontraron imágenes en el set de validación.")
        return

    # 3. Predicciones
    print("🔮 Generando predicciones (esto puede tardar)...")
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # 4. Matriz de Confusión
    print("📊 Generando Matriz de Confusión...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Ajuste de tamaño para que se vean bien las 120 clases (será grande)
    plt.figure(figsize=(20, 20)) 
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') # annot=False porque con 120 clases se satura el texto
    plt.title('Matriz de Confusión - AnimalNet (Validación)')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()

    # 5. Reporte
    print("\n📄 Reporte de Clasificación (Resumen):")
    # Imprimimos solo un resumen textual si son muchas clases, o el completo
    print(classification_report(y_true, y_pred, target_names=class_labels))

if __name__ == '__main__':
    evaluate()