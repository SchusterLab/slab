# Schuster Lab Instrument Drivers

## Usage

You need the following libraries

```bash
conda create env experiment_python2
source activate experiment_python2
pip install numpy matplotlib
pip install Pyro4==4.24
pip install Py
```

## Important

- This module should minimize dependency on slab datanalysis modules.
- Each instrument should be inside a sub folder. This makes it easy to attach a `README.md` for that instrument.
- Each instrument should have a `<instrument_name>_test.py` file.
   - this test file should have a function called `api_test(instrument_instance)` that tests the low-level apis.
   - behavioral tests should be in a second function, and should be separate from the low-level api tests.
   - a test `instrument.cfg` file is included in each folder, and excluded from version control in `.gitignore`, to allow developers to have their own instrument.cfg during development.
- Each instrument should have a `README.md` file. It should be sweet and useful.

## To Contribute:

The benefit of building instrument drivers this way, is that it allows us to have a README for each driver, and a consistent testing convention.

To help fix bugs and create new drivers, and help make everyone's life easier in the lab, I ask you the following:

1. Always write tests for your code.
2. Write up the README nicely, so that the first-year in your lab knows how to use this.

Many thanks to you future contributors in advance!

## [Helium Manifold](relaybox/heliummanifold.py)


