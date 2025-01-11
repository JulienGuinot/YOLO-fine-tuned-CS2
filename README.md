# 🎮 Détection d'Ennemis - Counter-Strike 2 🕹️

## 📜 Description
Ce projet utilise **YOLOv8** pour la détection en temps réel d'ennemis et alliés dans des images et vidéos issues de **Counter-Strike 2**. YOLO a été fine-tuné sur un dataset personnalisé de **4454 images annotées** provenant de parties de CS2.

## 🗃️ Dataset
Le dataset est disponible ici : [Dataset CS2](https://universe.roboflow.com/asd-culfr/wlots/dataset/1)

Le fichier `train_cs.py` permet d'entraîner le modèle et génère en sortie le modèle `best.pt`, qui peut être utilisé pour l'inférence. Vous pouvez également ré-entraîner le modèle ou utiliser directement le modèle `best.pt` !

## 🔍 Caractéristiques
- Détection en **temps réel** de 4 classes : `'ct'`, `'cthead'`, `'t'`, `'thead'` 🚨
- Support pour l'analyse d'**images** et de **vidéos** 🎥
- **GPU** supporté si disponible (CUDA + PyTorch) 💪

![batch test](/runs/detect/train2/train_batch2.jpg)



## 🛠️ Prérequis
- Python 3.x 🐍
- PyTorch ⚡
- OpenCV 🖼️
- Ultralytics YOLOv8 🔥

## 📦 Installation
1. Cloner le dépôt : 
   ```bash
   git clone https://github.com/JulienGuinot/YOLO-fine-tuned-CS2
   ```
2. Installer les dépendances : `pip install -r requirements.txt`
3. Exécuter le script d'inférence : `python inference.py`


## 🧑‍💻 Modèle

Le modèle est disponible dans le dossier models/ 🚀


## 🐞 Issues

Si vous rencontrez un problème lors de l'installation ou de l'inférence, n'hésitez pas à ouvrir une Issue sur le projet. Je ferai de mon mieux pour y répondre le plus rapidement possible 😊
