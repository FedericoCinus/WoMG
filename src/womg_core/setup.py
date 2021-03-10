import setuptools
from glob import glob


with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ['click==7.1.2', 'decorator==4.4.2', 'joblib==1.0.1',
                'networkx==2.5', 'numpy==1.19.5', 'scipy==1.5.4',
                'scikit-learn==0.24.1', 'tqdm==4.56.2']

def find_files(folder, rec=True):
    ''' Finds all pickle, csv, txt, nx files
    '''
    list_files = list(glob(str(folder)+'/**', recursive=rec))
    files = []
    for file in list_files:
        if file.endswith('txt') or file.endswith('csv') or file.endswith('pickle') or file.endswith('nx'):
            files.append(("/"+str(file[:file.rfind('/')]), [file]))
        else:
            continue
    return files

###############################################################################
setuptools.setup(
    name="womg-core",
    version="1.0.4",
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
    py_modules=['womgc'],
    entry_points='''
        [console_scripts]
        womgc=womg_core.__main__:main_cli
    ''',
    #scripts=['bin/womgc'],
    python_requires='>=3.8',
)
