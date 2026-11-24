# AnimalNet-Pro: Transfer Learning for 120 Dog Breeds🌟

 Project OverviewAnimalNet-Pro is a professional-grade image classification project built to demonstrate advanced Deep Learning techniques, modular software engineering practices, and robust MLOps deployment readiness. It specializes in classifying 120 distinct dog breeds using state-of-the-art Transfer Learning.This project showcases a complete workflow, from automated data cleaning and organization to live inference via a modern API.
 
 ### Key Highlights
 Model:  $\text{EfficientNetB0}$ (Pre-trained on ImageNet).

 Techniques: Transfer Learning (Freezing & Fine-Tuning), Data Augmentation, $\text{Class Weight}$ balancing.

 Performance: Achieved $84\%$ Validation Accuracy across 120 classes.

 > **Note:** this figure was measured with a validation generator that still applied data augmentation. The generator has been fixed (validation now uses clean images); the metric will be re-measured on the next training run.

 Deployment: $\text{FastAPI}$ server integrated with $\text{TensorFlow/Keras}$ for low-latency inference.
 
 📁 Project Structure
The code is organized into a modular structure for clarity, scalability, and data engineering:

Plaintext

animal_classifier/
├── api/
│   └── app.py            # FastAPI application for real-time predictions
├── config/
│   └── config.py         # Global variables (paths, hyperparameters, 120 CLASSES list)
├── data/
│   └── train/            # Clean structure: 120 breed folders with JPGs
├── models/
│   └── best_model.keras  # The trained model weights (EfficientNetB0 + Custom Head)
└── src/
    ├── data_loader.py    # Data generators & augmentation pipeline
    ├── model.py          # Model architecture & unfreezing logic
    ├── train.py          # Main training loop (Transfer Learning + Fine-Tuning)
    ├── evaluate.py       # Metrics, Confusion Matrix & Classification Report
    ├── data_setup.py     # ETL Script: Organizes raw datasets & updates config
    └── cleanup.py        # Utility: Recursively removes non-image metadata files

🚀 Getting Started
Follow these steps to reproduce the environment and run the project.

1. PrerequisitesYou must have Python 3.9+ installed.Bash
      1.- Clone the repository
      git clone [YOUR-REPO-URL]
      cd animal_classifier

      2.- Create and activate the virtual environment
      python -m venv venv
      .\venv\Scripts\activate  # Windows
      source venv/bin/activate # Linux/Mac

      3.- Install dependencies
      pip install -r requirements.txt

2. Data Setup (ETL Pipeline)
This project includes custom scripts to handle the Stanford Dogs Dataset raw structure.
Download: Obtain the images.tar from the Stanford Dogs Dataset.

Extract: Place the extracted Images folder (containing n0208... folders) in the project root.

Run ETL Script: This script moves folders to data/train, cleans the names (e.g., n0208...-Chihuahua $\rightarrow$ Chihuahua), and auto-updates config.py.Bashpython -m src.data_setup

(Optional) Cleanup: If the dataset contains metadata files (like .xml or 1KB files), run the cleanup utility:Bashpython -m src.cleanup

3. Training & EvaluationThe training process runs in two phases automatically: Feature Extraction (Frozen base) and Fine-Tuning (Unfrozen top layers).Bash

     1.- Start Training (Approx. 25-30 epochs total)
     python -m src.train
     2.- Evaluate the best model (Generates Confusion Matrix & Metrics)
     python -m src.evaluate

### 🌐 API Deployment (Inference)The project is designed for immediate deployment using $\text{FastAPI}$.
1. Launch the ServerExecute the following command from the root directory:Bashuvicorn api.app:app --reload
2. Test the EndpointDocumentation URL: Open your browser and navigate to http://127.0.0.1:8000/docs.Inference: Use the interactive interface to upload an image and receive a JSON response with the top prediction and probability vector.
JSON{
  "filename": "my_dog_image.jpg",
  "prediction": "Rottweiler",
  "confidence": 0.9876,
  "probabilities": {
    "Rottweiler": 0.9876,
    "German_shepherd": 0.0012,
    // ... 118 other classes
  }
}

### 💡 Future Work & Expansion 
The project architecture is optimized for domain expansion:Integration: To add new classes (e.g., Cats, Lions), simply add the new folders (data/train/cat/, data/train/lion/) and update the CLASSES list in config/config.py.Fine-Tuning: Rerunning python -m src.train will load the specialized dog weights and perform efficient Fine-Tuning on the combined dataset.Containerization: Implementation of a Dockerfile for streamlined cloud deployment.

### Contributions 🤝
Contributions are welcome! Please open an issue before submitting a pull request to discuss your proposal.

## Developed by Heriberto Ganesha Cortés Valdez. 
l25121393@morelia.tecnm.mx