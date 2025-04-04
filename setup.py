from setuptools import setup, find_packages

setup(
    name='pyscarcopula',
    version='0.1.0',
    description='Stochastc copula models for VaR and CVaR risk assessment', 
    author='Alexey Novokhatskiy', 
    author_email='aanovokhatskiy@gmail.com', 
    packages=find_packages(),
    package_data={'pyscarcopula': ['auxiliary/*', 'marginal/*', 'sampler/*', 'stattests/*'],
                  'pyscarcopula.cython_ext': ['*.pxd']},
    install_requires=[
        'numpy',
        'numba',
        'scipy',
        'sympy',
        'joblib',
    ]
)
