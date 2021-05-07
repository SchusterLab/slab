"""
setup.py - a module to allow package installation
"""

from distutils.core import setup

NAME = "slab"
VERSION = "0.1"
DEPENDENCIES = [
    "numpy",
    "scipy"
]
DESCRIPTION = "This package is used for Schuster Lab experiments"
AUTHOR = "David Schuster"
AUTHOR_EMAIL = "david.schuster@uchicago.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)
