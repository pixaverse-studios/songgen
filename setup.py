from setuptools import setup, find_packages

setup(
    name="songgen",
    version="0.1.0",
    description="A deep learning model for generating singing voice from lyrics and melody",
    author="Pixa Labs",
    author_email="nitish@heypixa.ai", 
    url="https://github.com/pixa-labs/songgen",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "omegaconf>=2.1.0",
        "pypinyin>=0.47.0",
        "hangul-romanize>=0.1.0",
        "flash-attn>=2.0.0",
        "num2words>=0.5.10",
        "spacy>=3.0.0",
        "demucs>=4.0.0",
        "descript-audiotools>=0.7.2",
        "descript-audio-codec"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update with actual license
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 