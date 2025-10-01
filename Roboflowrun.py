from roboflow import Roboflow

print("Connecting to Roboflow...")

rf = Roboflow(api_key="KZGgITtcYEN5EjhFFl62")

# Load your workspace and project
project = rf.workspace("thepbordin-8lgun").project("blind-assistant")
version = project.version(1)

# Download dataset in YOLOv8 format
dataset = version.download("yolov8")

print("Dataset downloaded successfully!")
