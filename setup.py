"""Legacy setup.py for editable installs: pip install -e ."""
from setuptools import setup, find_packages

setup(
    name="reli-turbo",
    version="0.1.0",
    description="GPU-accelerated RELI permutation testing for RBP enrichment analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["reli_turbo*"]),
    install_requires=[
        "cupy-cuda12x>=13.0",
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    entry_points={
        "console_scripts": [
            "reli-turbo=reli_turbo.cli:main",
        ],
    },
)
