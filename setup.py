import setuptools

setuptools.setup(
    name="nano_trf",
    version="0.1",
    author="Tyler Ganter",
    author_email="tyler.h.ganter@example.com",
    description="A package for learning about transformer architectures",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "jupyterlab",
        "numpy",
        "scikit-learn",
        "torch>=2.0",
        "requests",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
