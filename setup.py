import setuptools
from setuptools import setup

TEST_DEPS = [
    "pytest==7.4.0",
    "pytest-runner==6.0.0",
    "pytest-cov==4.1.0",
    "nox==2023.4.22",
]

API_DEPS = [
    "pydantic==2.1.1",
    "fastapi==0.101.0",
    "uvicorn"
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
    entry_points={"console_scripts": ["contact-reason=gorgias_ml.cli.cli:main"]},
    install_requires=requirements,
    extras_require={"test": TEST_DEPS,
                    "api": API_DEPS
                    }
)
