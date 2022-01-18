import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="flocpy",
    version="0.1.dev",
    description="Floc image processing tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Floc-Imaging-and-Image-Processing/flocpy",
    author="Thomas Ashley",
    author_email="tashley22@gmail.com",
    license="MIT",
    packages=["flocpy", "id", "stats"],
    include_package_data=True,
    install_requires=["scikit-image>=0.18.1", "numpy", "matplotlib", "pandas"
    "datetime", "tqdm", "joblib", "scipy", "statsmodels"]
)