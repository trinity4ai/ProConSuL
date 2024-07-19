import setuptools


setuptools.setup(
    name='proconsul',
    version='0.0.0',
    description='ProConSuL framework',
    packages=setuptools.find_packages(),
    install_requires=[],
    package_data={'proconsul.evaluation': ['test_datasets/*/*.*']},
    python_requires=">=3.11",
    entry_points="""
        [console_scripts]
        proconsul=proconsul.cli:cli
    """
)
