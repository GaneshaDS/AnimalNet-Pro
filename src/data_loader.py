import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from config.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR

def get_data_generators():
    """
    Generadores de datos robustos con Augmentation para Train.
    """
    # 1. Configuración de Augmentation (Solo para train)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # Normalización vital para EfficientNet
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # 20% automático para validación
    )

    # 2. Validación: mismo split que train, pero SIN augmentation.
    #    Keras reparte los archivos de forma determinista según validation_split,
    #    así que el subset 'validation' es el mismo que antes; solo cambia que
    #    las imágenes ya no se rotan/escalan al medir la métrica.
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    # 3. Generador de Test (Sin augmentation, solo preprocesamiento)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print(f"📥 Cargando datos desde: {TRAIN_DIR}")

    # Generador de Entrenamiento
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Generador de Validación
    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Generador de Test (Opcional)
    test_gen = None
    if os.path.exists(TEST_DIR):
        try:
            test_gen = test_datagen.flow_from_directory(
                TEST_DIR,
                target_size=IMG_SIZE,
                batch_size=1, # Batch 1 para evaluar una por una
                class_mode='categorical',
                shuffle=False # Importante NO mezclar para la matriz de confusión
            )
        except Exception as e:
            print(f"⚠️ Error cargando test: {e}")
    else:
        print("⚠️ Carpeta 'test' no encontrada. Se omitirá el set de prueba.")

    return train_gen, val_gen, test_gen
