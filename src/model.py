import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from config.config import IMG_SIZE

def build_model(num_classes):
    """
    Construye el modelo EfficientNetB0 con Transfer Learning.
    """
    # 1. Cargar base pre-entrenada (ImageNet)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False, # Sin la capa final de 1000 clases
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # 2. Congelar capas base (Feature Extraction)
    base_model.trainable = False 

    # 3. Cabezal de clasificación personalizado
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Nota: EfficientNet tiene su propia normalización interna, 
    # pero preprocess_input en data_loader ayuda.
    
    x = base_model(inputs, training=False) # training=False mantiene la BatchNorm estable
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x) # Regularización para evitar Overfitting
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="AnimalNet_EfficientB0")
    
    return model, base_model

def unfreeze_model(model, base_model, num_layers_to_unfreeze=20):
    """
    Descongela las últimas N capas para Fine-Tuning.
    """
    base_model.trainable = True
    
    # Congelar todas MENOS las últimas N
    # Se usa [:-N] para seleccionar desde el inicio hasta N antes del final
    count = 0
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
        count += 1
        
    print(f"🔓 Se han descongelado las últimas {num_layers_to_unfreeze} capas. ({count} capas permanecen congeladas).")
    return model