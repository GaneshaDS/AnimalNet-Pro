import os
from config.config import TRAIN_DIR

def delete_non_image_files(directory):
    count = 0
    # os.walk recorre el directorio de forma recursiva (subcarpetas)
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            
            # Buscamos y eliminamos cualquier archivo que NO termine en .jpg, .jpeg o .png
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"No se pudo eliminar {file_path}: {e}")
    return count

if __name__ == '__main__':
    print(f"--- 🗑️ Limpiando archivos no imagen en {TRAIN_DIR} ---")
    deleted_count = delete_non_image_files(TRAIN_DIR)
    print(f"✅ Limpieza finalizada. Se eliminaron {deleted_count} archivos de metadata.")