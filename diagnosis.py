def generate_diagnosis(results):
    diagnosis = []
    for det in results["detections"]:
        class_id = int(det[5])  # YOLO class index
        if class_id == 0:
            diagnosis.append("Possible Pneumonia detected")
        elif class_id == 1:
            diagnosis.append("Bone Fracture detected")
        else:
            diagnosis.append("Review required: Unclassified anomaly")
    return diagnosis if diagnosis else ["No anomaly detected"]
