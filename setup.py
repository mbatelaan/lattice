import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="lattice",
    version="0.0.1",
    author="Mischa Batelaan",
    author_email="mischa.batelaan@adelaide.edu.au",
    description="Lattice analysis functions",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url="https://github.com/AndreScaffidi/NatPy",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
