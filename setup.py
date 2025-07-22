from setuptools import setup, find_packages

setup(
    name="flow-ssn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "h5py>=3.11.0",
        "matplotlib>=3.8.0",
        "numpy>=2.3.1",
        "pandas>=2.3.1",
        "Pillow>=11.3.0",
        "scikit-learn>=1.4.2",
        "scipy>=1.16.0",
        "torch>=2.5.0",
        "torchdiffeq>=0.2.3",
        "torchvision>=0.20.0",
        "tqdm>=4.65.0",
        "wandb>=0.21.0",
    ],
    extras_require={"dev": ["black"]},
    python_requires=">=3.9",
)
