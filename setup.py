from setuptools import setup

setup(
    name="superneuromat",
    packages=["superneuromat"],
    version="0.2",
    author="Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti",
    author_email="datepa@ornl.gov",
    include_package_data=True,
    url="https://github.com/ORNL/superneuromat",
    download_url="https://github.com/ORNL/superneuromat/archive/refs/tags/v_01.tar.gz",
    keywords=["Neuromorphic Computing", "Neuromorphic Simulator", "Neuromorphic Algorithms", "Fast Neuromorphic Simulator", "Matrix-Based Neuromorphic Simulator", "Numpy Neuromorphic Simulator"],
    license="BSD 3-Clause",
    description="A matrix-based simulation framework for neuromorphic computing.",
    long_description="""A matrix-based simulation framework for neuromorphic computing.""",
    long_description_content_type="text/markdown",
    project_urls={"Source": "https://github.com/ORNL/superneuromat"},
    install_requires=["numpy", "pandas"],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD 3-Clause',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ],
)

