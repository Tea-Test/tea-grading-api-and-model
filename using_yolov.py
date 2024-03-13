from ultralytics import YOLO

# load the model
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# train the model
model.train(data='/home/warrior/Development/imageclassificationtutorial/tea_grades_dataset', epochs=20, imgsz=64)