from setuptools import setup

setup(
    name='sparsecca',
    version='0.1.0',
    description="",
    author='Erkka Heinila',
    author_email='erkka.heinila@jyu.fi',
    license='BSD',
    packages=['sparsecca'],
    keywords='pma cca pmd sparse svd',
    install_requires=[
        'setuptools',
        'pandas',
        'numpy',
        'statsmodels'
    ],
)
