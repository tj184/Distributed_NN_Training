from setuptools import setup, find_packages

setup(
    name='distributed_trainer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
    ],
    entry_points={
        'console_scripts': [
            'pc1-train=pc1_trainer:main',
            'pc2-train=pc2_trainer:main'
        ]
    }
)
