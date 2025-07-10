from setuptools import setup, Extension, find_packages
import os

compiled_ext = None
binary_name = None
for ext_name in ["compress.so", "compress.pyd"]:
    candidate = os.path.join("src", "tree_compress", ext_name)
    if os.path.isfile(candidate):
        compiled_ext = candidate
        binary_name = ext_name
        break

if compiled_ext is None:
    raise RuntimeError("Compiled binary not found! Run Nuitka build step first.")

ext_modules = [
    Extension(
        "tree_compress.compress",
        sources=[],  # Precompiled by Nuitka
    ),
]

setup(
    name="lop_compress",
    version="0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    package_data={
        "tree_compress": [binary_name],  # Include the exact binary found
    },
    include_package_data=True,
)
