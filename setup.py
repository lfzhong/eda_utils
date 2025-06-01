from setuptools import setup, find_packages

setup(
    name='eda_package',
    version='0.1',
    description='A Python package for exploratory data analysis',
    author='Fangzhong Liu',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn'
    ],
)
