from setuptools import setup, find_namespace_packages

setup(
    name="kservehelper",
    version="1.0.1",
    author="Wenzhuo Yang",
    description="A KServe Model Wrapper",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yangwenz/kserve-helper",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="kservehelper.*"),
    package_dir={"kservehelper": "kservehelper"},
    install_requires=[
        "kserve==0.10.2",
        "pydantic==1.10.12",
        "requests==2.29.0",
        "aiohttp==3.8.3",
        "aiofiles==23.2.1",
        "pytest"
    ],
    python_requires=">=3.8,<4",
    zip_safe=False,
)
