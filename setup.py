from setuptools import setup, find_namespace_packages

setup(
    name="kservehelper",
    version="1.2.4",
    author="Wenzhuo Yang",
    description="A KServe Model Wrapper",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yangwenz/kserve-helper",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="kservehelper.*"),
    package_dir={"kservehelper": "kservehelper"},
    entry_points={
        "console_scripts": [
            "kservehelper=kservehelper.cli:main",
        ]
    },
    install_requires=[
        "kserve==0.10.2",
        "pydantic==1.10.12",
        "requests==2.29.0",
        "aiohttp==3.8.3",
        "aiofiles==23.2.1",
        "schema",
        "pyyaml",
        "click",
        "pytest",
        "mmh3",
        "boto3"
    ],
    python_requires=">=3.8,<4",
    zip_safe=False,
)
