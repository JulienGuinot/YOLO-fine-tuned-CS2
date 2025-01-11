from ultralytics import YOLO
import cv2
import torch
import os

# Charger le modèle pré-entraîné YOLOv8
model = YOLO('yolov8n.pt')  # Charger YOLOv8n, le plus petit modèle

# Configurer et lancer l'entraînement
def train_model():
    # Configuration de l'entraînement
    results = model.train(
        data=os.path.abspath('wlots.v1i.yolov11/data.yaml'),  # utiliser le chemin absolu
        epochs=100,  # nombre d'époques
        imgsz=640,  # taille des images
        batch=16,  # taille du batch
        patience=50,  # nombre d'époques sans amélioration avant de stopper l'entraînement
        device='0' if torch.cuda.is_available() else 'cpu'  # utiliser GPU si disponible
    )
    
    return results

# Pour une image
def detect_image(image_path):
    results = model.predict(image_path, conf=0.25)
    for r in results:
        im_array = r.plot()  # afficher les résultats sous forme de graphique
        cv2.imshow("Détection", im_array)
        cv2.waitKey(0)

# Pour une vidéo
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.predict(frame, conf=0.25)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Détection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(f"Utilisation du device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    train_model()