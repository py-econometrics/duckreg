from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

def read_requirements():
    try:
        with open(os.path.join(here, "requirements.txt"), "r") as req:
            return req.read().splitlines()
    except FileNotFoundError:
        return []

def read_long_description():
    try:
        with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

setup(
    name="duckreg",
    version="0.2",  # Use semantic versioning
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Apoorva Lal",
    author_email="lal.apoorva@gmail.com",
    description="A package for Regression in compressed representation powered by DuckDB",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="statistics, econometrics, sufficient statistics, bootstrap",
    url="https://github.com/apoorvalal/duckreg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'duckreg': ['*.py'],
    },
)
