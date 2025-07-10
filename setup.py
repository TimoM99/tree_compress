from setuptools import setup, Extension, find_packages
import os

# Find the compiled binary extension
compiled_ext = None
for ext_name in ["compress.so", "compress.pyd"]:
    candidate = os.path.join("src", "tree_compress", ext_name)
    if os.path.isfile(candidate):
        compiled_ext = candidate
        break

if compiled_ext is None:
    raise RuntimeError("Compiled binary not found! Run Nuitka build step first.")

ext_modules = [
    Extension(
        "tree_compress.compress",
        sources=[],  # Empty because binary already compiled by Nuitka
    ),
]

setup(
    name="lop_compress",
    version="0.3",  # Keep in sync with pyproject.toml or consider centralizing
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    package_data={
        "tree_compress": ["compress.so", "compress.pyd"],  # Include whichever exists
    },
    # Metadata here if you want, but pyproject.toml handles most of it
)
