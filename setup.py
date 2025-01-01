from setuptools import setup, find_packages
import os

# Read README.md if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="dgNova",
    version="1.2.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "Pillow>=8.0.0",
        "openpyxl>=3.0.0"
    ],
    extras_require={
        'animation': [
            'ffmpeg-python',
            'imagemagick'
        ]
    },
    author="Nadim Khan",
    author_email="nfornadim@gmail.com",
    description="Statistical Analysis and Simulations for Plant Breeding Experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nfornadimkhan/dgNova",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 