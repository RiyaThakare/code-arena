from ultralytics import YOLO
import torch
import cv2

# Load pretrained models
yolo_model = YOLO("models/yolov8/best.pt")  
unet_model = torch.load("models/unet/unet_weights.pth", map_location="cpu")
maskrcnn_model = torch.load("models/maskrcnn/maskrcnn_weights.pth", map_location="cpu")

def run_inference(image_path):
    # Run YOLOv8 detection
    yolo_results = yolo_model.predict(image_path, save=True, conf=0.3)
    annotated_path = yolo_results[0].save_dir / yolo_results[0].path.name

    # TODO: Add U-Net + Mask R-CNN segmentation inference
    # For now, keep YOLO as base output
    results = {
        "detections": yolo_results[0].boxes.data.cpu().numpy(),
        "annotated_path": str(annotated_path)
    }
    return results
