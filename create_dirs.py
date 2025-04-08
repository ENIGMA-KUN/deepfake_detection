import os

directories = [
    "data",
    "data/raw",
    "data/processed",
    "data/interim",
    "models",
    "models/image",
    "models/audio",
    "models/video",
    "src",
    "src/data",
    "src/features",
    "src/models",
    "src/models/image",
    "src/models/audio",
    "src/models/video",
    "src/visualization",
    "src/api",
    "notebooks",
    "logs",
    "tests",
    "docs"
]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")