import os
import shutil
import re
from config.config import DATA_DIR, TRAIN_DIR

def clean_and_organize_dataset(raw_data_path):
    """
    Actúa como una API interna para limpiar y mover las carpetas de razas 
    (ej. n0208...-papillon) a la estructura de entrenamiento (data/train/papillon).
    """
    print("--- 📂 Iniciando preparación del dataset ---")
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Error: Ruta de datos brutos no encontrada: {raw_data_path}")
        print("Asegúrate de que la ruta sea correcta y la carpeta exista.")
        return []

    # Crear la carpeta de destino si no existe
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    new_classes = []
    
    # 1. Iterar sobre los directorios en la carpeta raw ('Images' en este caso)
    for item_name in os.listdir(raw_data_path):
        raw_class_path = os.path.join(raw_data_path, item_name)
        
        # Verificar si es un directorio y si tiene el formato n0208...
        if os.path.isdir(raw_class_path) and item_name.startswith('n0'):
            
            # 2. Limpiar el nombre (ej: 'n02086910-papillon' -> 'papillon')
            # El split('-') y [-1] garantiza que solo se tome el nombre de la raza.
            # Nota: Si estás moviendo la carpeta 'Images', esta parte del nombre 
            # de carpeta puede ser diferente a 'Annotation'. 
            # ASUMIENDO que la carpeta 'Images' usa la misma convención de nombres:
            parts = item_name.split('-')
            clean_name = parts[-1] if len(parts) > 1 else item_name # Maneja si no hay guion

            # Caso especial: Si estás usando la carpeta 'Images', a veces las subcarpetas 
            # ya NO tienen el guion y solo tienen la nomenclatura n0208....
            # Si ese es tu caso, deberías obtener los nombres limpios de otro lado 
            # (ej. de la lista de clases en config.py), pero por ahora, mantendremos la 
            # lógica original, que asume que el nombre de la raza está después del guion.
            
            target_class_path = os.path.join(TRAIN_DIR, clean_name)
            
            print(f"   -> Moviendo y renombrando: {item_name} -> {clean_name}")

            try:
                # 3. Mover la carpeta completa a la estructura de training
                shutil.move(raw_class_path, target_class_path)
                new_classes.append(clean_name)
            except Exception as e:
                print(f"⚠️ Error al mover {item_name}: {e}")
                
        elif os.path.isdir(raw_class_path) and item_name.lower() in ['annotation', 'readme', 'licenses']:
            print(f"   -> Omitiendo carpeta auxiliar: {item_name}")
            
        # Agregamos esta excepción para que no mueva la carpeta de imágenes si no tiene el formato n0208
        elif os.path.isdir(raw_class_path) and item_name.lower() == 'images':
            print(f"   -> Omitiendo carpeta auxiliar: {item_name}")