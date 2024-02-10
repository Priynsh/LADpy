import pathlib
import setuptools

setuptools.setup(
    name="binpat",
    version="0.1.0",
    description="A powerful tool for finding patterns with binary data",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type = "text/markdown",
    url="",
    author="Priyansh Jain",
    author_email="priyansh.contact@gmail.com",
    license="MIT",
    project_urls={
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows :: Windows 11',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires = "",
    install_requires = [],
    packages = setuptools.find_packages(),
    include_package_data=True
)