from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="EasyNMT",
    version="2.0.2",
    author="Nils Reimers",
    author_email="info@nils-reimers.de",
    description="Easy to use state-of-the-art Neural Machine Translation",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/UKPLab/EasyNMT",
    download_url="https://github.com/UKPLab/EasyNMT/archive/v2.0.2.zip",
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'transformers>=4.4,<5',
        'torch>=1.6.0',
        'numpy',
        'nltk',
        'sentencepiece',
        'fasttext',
        'protobuf'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Neural Machine Translation"
)