from ultralytics import YOLO

# Load the pretrained YOLOv8 model
model = YOLO("yolov8s.pt")

# Perform object detection on the image
results = model("images/test.jpeg")

# Save the annotated image
results[0].save(filename="output/result.jpeg")

print("Object detection completed successfully!")