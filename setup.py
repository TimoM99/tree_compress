from setuptools import setup, find_packages

setup(
    name="lop_compress",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)