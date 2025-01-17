from setuptools import setup, find_packages

setup(
    name='airsoul',  
    version=airsoul.__version__,  
    packages=find_packages(),  
    package_dir={'': '.'},  
    install_requires=[
        'numpy>=1.18.0',
        'torch>=1.13.0', 
        'restools>=0.0.0.10',
        'mamba-ssm==2.2.2',
        'fla@git+https://github.com/WorldEditors/flash-linear-attention.git@dev'
    ],
)