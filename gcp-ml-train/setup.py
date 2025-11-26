from setuptools import setup, find_packages

setup(
    name='gcp-ml-train',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A machine learning training package for Google Cloud Platform',
    packages=find_packages(where='trainer'),
    package_dir={'': 'trainer'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'tensorflow',  # or 'torch' if using PyTorch
        'optuna',
        'joblib',
        'matplotlib',
        'seaborn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)