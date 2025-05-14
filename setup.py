from setuptools import setup, find_packages

setup(
    name="spectral_subtraction",
    version="1.0",
    packages=find_packages(include=["scripts", "scripts.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "argparse",
        "sms-tools",
        "matplotlib",
        "librosa",
        "IPython",
    ],
    entry_points={
        "console_scripts": [
            "spectral-subtraction = scripts.main:main"  
        ]
    }
)
