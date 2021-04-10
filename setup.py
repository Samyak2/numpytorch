import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="numpytorch",  # Replace with your own username
    version="0.0.1",
    author="Samyak S Sarnayak",
    author_email="samyak201@gmail.com",
    description="Simple neural network implementation in numpy with a PyTorch-like API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Samyak2/numpytorch",
    packages=setuptools.find_packages(exclude=["examples"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
