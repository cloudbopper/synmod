"""synmod package definition and install configuration"""

from setuptools import find_packages, setup

# pylint: disable = invalid-name

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

setup(
    author="Akshay Sood",
    author_email="sood.iitd@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description="Temporal model synthesis",
    entry_points={
        "console_scripts": [
            "synmod=synmod.master:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "mihifepe",
        "numpy",
    ],
    keywords="synmod",
    license="MIT",
    long_description=readme + "\n\n" + changelog,
    name="synmod",
    packages=find_packages(include=["synmod"]),
    python_requires=">= 3.5",
    url="https://github.com/cloudbopper/synmod",
    version="0.1.1",
    zip_safe=False,
)
