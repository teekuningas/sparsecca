from setuptools import setup

setup(
    name='pypma',
    version='0.1.0',
    description="Implements Witten et al. 2009 in python",
    author='Erkka Heinila',
    author_email='erkka.heinila@jyu.fi',
    license='BSD',
    packages=['pypma'],
    keywords='pma cca pmd svd',
    install_requires=[
        'setuptools',
    ],
)
