"""
    Package dependencies and versions
"""
import setuptools

NAME = 'trainer'
VERSION = '1.0'
REQUIRED_PACKAGES = [
    'tensorflow-gpu==1.14.0',
    'google-cloud-storage'
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='Image classification training application package')
