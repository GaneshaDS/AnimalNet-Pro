import tensorflow as tf
import numpy as np
import os
from sklearn.utils import class_weight
from src.data_loader import get_data_generators
from src.model import build_model, unfreeze_model
from config.config import EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH

def train():
    print("🚀 Iniciando proceso de entrenamiento...")
    
    # 1. Obtener datos
    train_gen, val_gen, _ = get_data_generators()
    
    if train_gen.samples == 0:
        print("❌ Error: No se encontraron imágenes de entrenamiento. Revisa la carpeta data/train.")
        return

    num_classes = train_gen.num_classes
    print(f"📊 Clases detectadas: {list(train_gen.class_indices.keys())}")

    # 2. Calcular Class Weights (Para datasets desbalanceados)
    print("⚖️ Calculando pesos de clases...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"⚖️ Pesos: {class_weights_dict}")

    # 3. Construir Modelo
    model, base_model = build_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6, monitor='val_loss', verbose=1),
        # Guardamos en .keras explícitamente
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    # 5. Fase 1: Entrenamiento del Clasificador (Transfer Learning)
    print("\n🏁 --- FASE 1: Entrenando cabezal ---")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # 6. Fase 2: Fine-Tuning
    print("\n🔧 --- FASE 2: Fine-Tuning ---")
    model = unfreeze_model(model, base_model, num_layers_to_unfreeze=30)
    
    # Recompilar con LR muy bajo (10x menor)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_gen,
        epochs=10, # Épocas extra
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    print(f"\n✅ Modelo guardado exitosamente en: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()