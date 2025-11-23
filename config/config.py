import os

# --- Rutas Absolutas (Agnósticas al SO) ---
# BASE_DIR es la carpeta 'animal_classifier'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Crear directorio de modelos si no existe
os.makedirs(MODELS_DIR, exist_ok=True)

# Ruta del modelo final (Formato .keras moderno)
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'best_model.keras')

# --- Hiperparámetros ---
IMG_SIZE = (224, 224)  # Tamaño nativo para EfficientNetB0
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Ajusta esto según las carpetas que tengas en data/train
# En config/config.py

# ... (otras configuraciones)

# AJUSTE AUTOMÁTICO DE LAS 120 CLASES DE PERROS:
CLASSES = [
           'Chihuahua',
           'Japanese_spaniel',
           'Maltese_dog',
           'Pekinese',
           'Shih-Tzu',
           'Blenheim_spaniel',
           'papillon',
           'toy_terrier',
           'Rhodesian_ridgeback',
           'Afghan_hound',
           'basset',
           'beagle',
           'bloodhound',
           'bluetick',
           'black-and-tan_coonhound',
           'Walker_hound',
           'English_foxhound',
           'redbone',
           'borzoi',
           'Irish_wolfhound',
           'Italian_greyhound',
           'whippet',
           'Ibizan_hound',
           'Norwegian_elkhound',
           'otterhound',
           'Saluki',
           'Scottish_deerhound',
           'Weimaraner',
           'Staffordshire_bullterrier',
           'American_Staffordshire_terrier',
           'Bedlington_terrier',
           'Border_terrier',
           'Kerry_blue_terrier',
           'Irish_terrier',
           'Norfolk_terrier',
           'Norwich_terrier',
           'Yorkshire_terrier',
           'wire-haired_fox_terrier',
           'Lakeland_terrier',
           'Sealyham_terrier',
           'Airedale',
           'cairn',
           'Australian_terrier',
           'Dandie_Dinmont',
           'Boston_bull',
           'miniature_schnauzer',
           'giant_schnauzer',
           'standard_schnauzer',
           'Scotch_terrier',
           'Tibetan_terrier',
           'silky_terrier',
           'soft-coated_wheaten_terrier',
           'West_Highland_white_terrier',
           'Lhasa',
           'flat-coated_retriever',
           'curly-coated_retriever',
           'golden_retriever',
           'Labrador_retriever',
           'Chesapeake_Bay_retriever',
           'German_short-haired_pointer',
           'vizsla',
           'English_setter',
           'Irish_setter',
           'Gordon_setter',
           'Brittany_spaniel',
           'clumber',
           'English_springer',
           'Welsh_springer_spaniel',
           'cocker_spaniel',
           'Sussex_spaniel',
           'Irish_water_spaniel',
           'kuvasz',
           'schipperke',
           'groenendael',
           'malinois',
           'briard',
           'kelpie',
           'komondor',
           'Old_English_sheepdog',
           'Shetland_sheepdog',
           'collie',
           'Border_collie',
           'Bouvier_des_Flandres',
           'Rottweiler',
           'German_shepherd',
           'Doberman',
           'miniature_pinscher',
           'Greater_Swiss_Mountain_dog',
           'Bernese_mountain_dog',
           'Appenzeller',
           'EntleBucher',
           'boxer',
           'bull_mastiff',
           'Tibetan_mastiff',
           'French_bulldog',
           'Great_Dane',
           'Saint_Bernard',
           'Eskimo_dog',
           'malamute',
           'Siberian_husky',
           'affenpinscher',
           'basenji',
           'pug',
           'Leonberg',
           'Newfoundland',
           'Great_Pyrenees',
           'Samoyed',
           'Pomeranian',
           'chow',
           'keeshond',
           'Brabancon_griffon',
           'Pembroke',
           'Cardigan',
           'toy_poodle',
           'miniature_poodle',
           'standard_poodle',
           'Mexican_hairless',
           'dingo',
           'dhole',
           'African_hunting_dog'
]