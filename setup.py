
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="scs4onnx",
    version="1.0.6",
    description="A very simple tool that compresses the overall size of the ONNX model by aggregating duplicate constant values as much as possible. Simple Constant value Shrink for ONNX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/scs4onnx",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "scs4onnx=scs4onnx:main"
        ]
    }
)
