import setuptools
from glob import glob
import pathlib
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements=['Click==7.0', 'networkx==2.3', 'numpy==1.17.3', 'scikit-learn==0.21.3', 'tqdm==4.32.1', 'gensim==3.8.0']


def find_files(folder, rec=True):
    ''' Finds all pickle, csv, txt, nx files
    '''
    list_files = list(glob(str(folder)+'/**', recursive=rec))
    files = []
    for file in list_files:
        if file.endswith('txt') or file.endswith('csv') or file.endswith('pickle') or file.endswith('nx'):
            files.append( ("/"+str(file[:file.rfind('/')]), [file]) )
        else:
            continue
    return files

###############################################################################
setuptools.setup(
    name="womg",
    version="0.9.1",
    author="Federico Cinus",
    author_email="federico.cinus@isi.it",
    description="Word-of-Mouth cascades Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FedericoCinus/WoMG.git",
    packages=setuptools.find_packages(),
    #include_package_data=True,
    #package_data = {'data': ['data/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files   = find_files('womgdata'),
    install_requires=requirements,
    py_modules=['womg'],
    entry_points='''
        [console_scripts]
        womg=womg.__main__:main_cli
    ''',
    #scripts=['bin/womg'],
    python_requires='>=3.2',
)
