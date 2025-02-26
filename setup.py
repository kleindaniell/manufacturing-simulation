from setuptools import setup, find_packages

setup(
    name="shop-environment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "simpy"
        "numpy",  # List dependencies
        "pandas",
    ]
)