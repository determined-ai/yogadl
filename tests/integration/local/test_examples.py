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
import pathlib
import runpy


def examples_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).parent
    return here.parent.parent.parent.joinpath("examples")


def test_walkthrough() -> None:
    runpy.run_path(str(examples_dir().joinpath("walkthrough.py")))


def test_mnist() -> None:
    runpy.run_path(str(examples_dir().joinpath("mnist.py")))


def test_custom_data_ref() -> None:
    runpy.run_path(str(examples_dir().joinpath("custom_data_ref.py")))
