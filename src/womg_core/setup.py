import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="womg-core", # Replace with your own username
    version="0.0.1",
    author="Federico Cinus",
    author_email="federico.cinus@isi.it",
    description="Word-of-Mouth cascades Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github....",
    packages=setuptools.find_packages(include=['womg_core.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #entry_points={'console_scripts': ['womg=womg_core.womg:womg_main']},
    scripts=['bin/womgc'],
    python_requires='>=3.6',
)
