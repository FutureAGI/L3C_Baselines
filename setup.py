from setuptools import setup, find_packages

setup(
    name='l3c_baselines',  
    version='0.1.0',  
    packages=find_packages(),  
    package_dir={'': '.'},  
    install_requires=[
        'numpy>=1.18.0',
        'torch>=1.13.0', 
        'restools>=0.0.0.8',
        'mamba-ssm==2.2.2'
    ],
    setup_requires=[
        'git+https://github.com/sustcsonglin/flash-linear-attention'
    ],
)
