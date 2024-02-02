from setuptools import setup, find_packages

setup(
    name='ssl_dd',
    version='0.1',
    description='Self-Supervised Learning for Damage Detection',
    author='Yacine Bel-Hadj',
    author_email='Yacine.bel-hadj@vub.be',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'lightning',
        'wandb',
        'numpy',
        'hydra-core',
        'pyrootutils',
        'pynvml',
        'hydra-colorlog',],
    python_requires='>=3.6 , <3.12',
)


