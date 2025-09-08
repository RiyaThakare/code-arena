from ultralytics import YOLO
import cv2
import os

model = YOLO("models/yolov8_model/best.pt")  # your trained model

def run_yolov8(image_path, output_folder):
    results = model.predict(source=image_path, save=False)
    img = cv2.imread(image_path)
    
    diagnosis_counts = {"Fracture":0, "Pneumonia":0, "Other":0}
    detections = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        cls = int(cls)
        if cls == 0:
            diag = "Fracture"
        elif cls == 1:
            diag = "Pneumonia"
        else:
            diag = "Other"

        diagnosis_counts[diag] += 1
        detections.append({"class": diag, "bbox": [x1,y1,x2,y2], "score": score})

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(img, diag, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    annotated_file = "annotated_" + os.path.basename(image_path)
    annotated_path = os.path.join(output_folder, annotated_file)
    cv2.imwrite(annotated_path, img)

    return annotated_file, diagnosis_counts, detections

