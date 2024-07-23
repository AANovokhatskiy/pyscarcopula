from setuptools import setup, find_packages

setup(
    name='pyscarcopula',
    version='0.1.0',
    packages=find_packages(include=['pyscarcopula', 'pyscarcopula.*']),
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'sympy',
        'tqdm',
        'joblib'
    ]
)
