from ultralytics import YOLO

    # Load a model
model = YOLO("yolo11n.pt")

    # Train the model
train_results = model.train(
        data="config.yaml",  # path to dataset YAML
        epochs=100
)
