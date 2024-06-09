import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="fedtreeavg/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton==2.0.0")

setup(
    name="fed-treeavg",
    py_modules=["fedtreeavg"],
    version=read_version(),
    description="FedTreeAvg",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="SuJeong Hur, Yonsei University",
    url="https://github.com/sdoubleoj/FedTreeAvg.git",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)