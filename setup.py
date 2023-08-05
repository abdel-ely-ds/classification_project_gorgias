import setuptools
from setuptools import setup

TEST_DEPS = [
    "pytest==7.4.0",
    "pytest-runner==6.0.0",
    "pytest-cov==4.1.0",
    "nox==2023.4.22",
]

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="gorgias_ml",
    version="0.1.0",
    description="Predicting contact reasons",
    keywords=["classification", "multi-class", "few shot learning"],
    author="abdel.ely.ds",
    license="MIT",
    classifiers=["Programming Language :: Python :: 3.10"],
    zip_safe=True,
    include_package_data=True,
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
    tests_require=TEST_DEPS,
    extras_require={"test": TEST_DEPS},
)
