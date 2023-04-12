from setuptools import setup

setup(
    name="superneuromat",
    version="0.1.0",
    author="Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti",
    author_email="datepa@ornl.gov",
    packages=["superneuromat"],
    include_package_data=True,
    url="https://github.com/ORNL/superneuromat",
    license="BSD 3-Clause",
    description="A matrix-based simulation framework for neuromorphic computing.",
    long_description="""A matrix-based simulation framework for neuromorphic computing.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://github.com/ORNL/superneuromat"},
    install_requires=["numpy", "pandas"],
)

