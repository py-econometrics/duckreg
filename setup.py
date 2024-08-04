from setuptools import setup, find_packages


# Function to read the contents of the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as req:
        return req.read().splitlines()


setup(
    name="duckreg",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
    # Additional metadata about your package
    author="Apoorva Lal",
    author_email="lal.apoorva@gmail.com",
    description="A package for Regression in compressed representation powered by DuckDB",
    license="MIT",
    keywords="statistics, econometrics, sufficient statistics, bootstrap",
    url="https://github.com/apoorvalal/duckreg",
)
