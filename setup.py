from setuptools import setup, find_packages

setup(
    name="deepfake_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "opencv-python",
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
    ],
) 