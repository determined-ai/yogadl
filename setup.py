# Copyright 2020 Determined AI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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
        "websockets",
    ],
    extras_require={
        "tf": ["tensorflow"]
    }
)
