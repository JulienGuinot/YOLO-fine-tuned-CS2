from ultralytics import YOLO
import cv2

# Charger le modèle entraîné
model = YOLO('models/best.pt')

# Pour une image
def detect_image(image_path):
    results = model.predict(image_path, conf=0.25)
    for r in results:
        im_array = r.plot()  # plot les résultats
        cv2.imshow("Détection", im_array)
        cv2.waitKey(0)

# Pour une vidéo avec sauvegarde
def detect_video(video_path, output_path):
    # Ouvrir la vidéo source
    cap = cv2.VideoCapture(video_path)
    
    # Obtenir les propriétés de la vidéo source
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurer le writer pour la sortie
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Faire la prédiction
            results = model.predict(frame, conf=0.25)
            annotated_frame = results[0].plot()
            
            # Afficher le frame
            cv2.imshow("YOLOv8 Détection", annotated_frame)
            
            # Sauvegarder le frame
            out.write(annotated_frame)
            
            # Quitter si 'q' est pressé
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()




