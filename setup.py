import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="alltools",
    version="0.0.1",
    author="Alexey Zabolotniy",
    author_email="alexey.zabolotniy.main@yandex.ru",
    description="All the neccessary stuff",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1101AlexZab1011/alltools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)