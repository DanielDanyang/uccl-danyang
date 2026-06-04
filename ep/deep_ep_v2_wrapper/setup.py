from setuptools import find_packages, setup


setup(
    name="deep_ep",
    version="2.0.0+ucclaws",
    packages=find_packages(),
    install_requires=["uccl"],
    author="uccl",
    description="DeepEP V2-compatible wrapper backed by UCCL-style AWS EFA proxy transport",
    python_requires=">=3.10",
)

