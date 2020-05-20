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
import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="yogadl",
    version="0.1.1",
    author="Determined AI",
    author_email="hello@determined.ai",
    url="https://www.github.com/determined-ai/yogadl/",
    description="Yoga Data Layer, a flexible data layer for machine learning",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires=">=3.6.0",
    install_requires=[
        "async_generator",
        "boto3",
        "filelock",
        "google-cloud-storage",
        "lmdb",
        "lomond",
        "websockets",
    ],
    extras_require={"tf": ["tensorflow"]},
    zip_safe=False,
    include_package_data=True,
)
