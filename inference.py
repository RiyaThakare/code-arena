from ultralytics import YOLO
import cv2
import os

model = YOLO("models/yolov8_model/best.pt")  # your trained model

def run_yolov8(image_path, output_folder):
    # Lower confidence threshold so we don’t miss detections
    results = model.predict(source=image_path, save=False, conf=0.1)
    img = cv2.imread(image_path)
    diagnosis = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        cls = int(cls)

        # ⚠️ Make sure this mapping matches your dataset preprocessing!
        if cls == 0:
            diagnosis.append("Pneumonia")
        elif cls == 1:
            diagnosis.append("Fracture")
        else:
            diagnosis.append("Unknown anomaly")

        # Draw bounding boxes with labels + confidence
        label = f"{diagnosis[-1]} ({score:.2f})"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    annotated_file = "annotated_" + os.path.basename(image_path)
    annotated_path = os.path.join(output_folder, annotated_file)
    cv2.imwrite(annotated_path, img)

    # Remove duplicates, but keep order
    unique_diagnosis = list(dict.fromkeys(diagnosis))

    return annotated_file, unique_diagnosis if unique_diagnosis else ["No anomaly detected"]
