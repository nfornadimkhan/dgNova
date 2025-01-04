from setuptools import setup, find_packages
import os

# Read README.md if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="dgNova",
    version="1.2.12",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
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
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
) 