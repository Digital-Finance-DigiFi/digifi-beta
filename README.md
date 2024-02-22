# DigiFi
General-purpose financial library. This package provides basic functions and algorithms used in finance.

## Disclaimer
Note that this package is not intended for commercial use, and the developers of the package do not accept any responsibility or liability for the accuracy or completeness of the code or the information provided.

## About The Project
We are a team of two developers who are enthusiastic about making a simple financial library for Python, which is powerful and optimized, yet minimalistic and easy to use. We intend the library to be open source for anyone to verify our code and/or propose improvements and additions to the future versions.

As of right now, we opened the code on [GitHub](https://github.com/Digital-Finance-DigiFi/digifi-beta) to get some early validation for the project. The current code base is implemented in Python, but in the future we aim to use Rust to do the "heavy lifting" and provide Python bindings for the new version of the library. The Python package will be available on [PyPI](https://pypi.org/), while we also want to make the Rust crate available on [Crates.io](https://crates.io/).

We aim to release the Beta version of the library implemented in Python, but we will not actively maintain it. Instead, we will try to address all the feedback and comments in the Rust implementation. So if you see any issues in the Python code please do report them, and we will try to fix these, but out main priority will be to provide a stable version of the library in Rust.

Separately to the main library, we are also working on a plotting package that provides plotting functions that go together with the functionality of the main library. The reason we are developing it as a standalone project is because we would like to keep the number of dependencies and the size of the main library down, such that it can be used in a very minimalistic setting (Currently, the Beta version only depends on [SciPy](https://scipy.org/) and [NumPy](https://numpy.org/)).

We will appreciate it if you are willing to help us debug and optimize the code, propose new functionality, or want to contribute in other ways, please do reach out to us!

DigiFi