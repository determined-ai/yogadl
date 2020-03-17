#!/usr/bin/env python3

from distutils.core import setup

setup(
    python_requires=">=3.6.0",
    version="0.1",
    name="yogadl",
    description="Yoga Data Layer, a flexible data layer for machine learning",
    author="Determined AI",
    author_email="info@determined.ai",
    url="https://www.github.com/determined-ai/yogadl/",
    packages=["yogadl"],
    install_requires=[
        "async_generator",
        "boto3",
        "filelock",
        "google-cloud-storage",
        "lmdb",
        "lomond",
        "tensorflow",
        "websockets",
    ],
)
