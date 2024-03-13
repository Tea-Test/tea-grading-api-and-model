from ultralytics import YOLO


def model_train():
    model = YOLO('/home/warrior/Development/tea-grading-api-and-model/runs/classify/train/weights/best.pt')
    return model