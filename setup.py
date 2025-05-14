from setuptools import setup, find_packages

setup(
    name="spectral_subtraction",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "argparse",
        "sms-tools",
        "matplotlib",
        "librosa",
    ],
    entry_points={
        "console_scripts": [
            "spectral-subtraction = main:main"
        ]
    }
)
