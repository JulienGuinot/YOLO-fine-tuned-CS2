# Détection d'enemis Counter-Strike 2

## Description
Ce projet utilise YOLOv8 pour la détection d'enemis/alliés dans des images et vidéos. YOLO a été fine-tuné sur un dataset personnalisé contenant 4454 images annotées, provenant de parties de Counter-Strike 2.

## Dataset
Le dataset se trouve ici `https://universe.roboflow.com/asd-culfr/wlots/dataset/1`
Le fichier train_cs.py permet d'entraîner le modèle, et donne en output le modèle `best.pt`, 
utilisable pour l'inférence. Vous pouvez ré-entrainer le modèle ou utiliser le modèle `best.pt`

## Caractéristiques
- Détection en temps réel de 4 classes : 'ct', 'cthead', 't', 'thead'
- Support pour l'analyse d'images et de vidéos
- Utilisation du GPU si disponible (cuda + pytorch)

## Prérequis
- Python 3.x
- PyTorch
- OpenCV
- Ultralytics YOLOv8

## Installation
1. Cloner le dépôt : `git clone https://github.com/JulienGuinot/YOLO-fine-tuned-CS2`
2. Installer les dépendances : `pip install -r requirements.txt`
3. Télécharger le modèle entraîné : `yolo export model=best.pt format=onnx`
4. Exécuter le script d'inférence : `python inference.py`


## Modèle
Le modèle est disponible dans le dossier `models/` 🚀 