import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='multiprocesser',
    version="0.1",
    packages=['multiprocesser'],
    url='',
    license='',
    author='erolsson',
    author_email='erolsson@kth.se',
    description=''
)
