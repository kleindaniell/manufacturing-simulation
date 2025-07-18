from setuptools import setup, find_packages

with open("requeriments.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="manusim",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "manusim=manusim.main:main",
        ],
    },
)
