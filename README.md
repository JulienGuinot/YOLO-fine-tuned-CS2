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
Le modèle est disponible dans le dossier `models/`
Ultralytics 8.3.58 🚀 Python-3.12.6 torch-2.5.1+cu121 CUDA:0 (NVIDIA GeForce RTX 2080 SUPER, 8192MiB)
Model summary (fused): 168 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 12/12 [00:02<00:00,  4.79it/s]
                   all        383        831      0.926      0.884      0.932      0.553
                    ct        189        244      0.926      0.927      0.953      0.593
                cthead        185        237      0.899      0.835      0.888      0.422
                     t        137        183      0.947      0.891      0.964       0.69
                 thead        123        167      0.931      0.884      0.923      0.508
Speed: 0.2ms preprocess, 1.5ms inference, 0.0ms loss, 0.9ms postprocess per image