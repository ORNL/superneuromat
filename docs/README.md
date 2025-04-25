To build the documentation, ensure you have the documentation-specific dependencies:

```bash
cd superneuromat
pip install -e .[docs]
```

```bash
cd docs
```

If you plan to upload the documentation to https://github.com/kenblu24/superneuromat-docs, you will need to clone the repository:

* ssh: `git clone git@github.com:kenblu24/superneuromat-docs.git`
* https: `git clone https://github.com/kenblu24/superneuromat-docs.git`

Then, from the `docs/` directory, run the following command to build the documentation:

```bash
./single clean
./single html -E
```

The `-E` forces `sphinx-build` to read all files, even if they have not changed.

For other options, see the [sphinx-build documentation](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#makefile-options).

The documentation will be built in the `docs/superneuromat-docs/build` directory,
and the homepage will be locally available at 
`docs/superneuromat-docs/build/html/index.html`.

The make interface is also available to make multiple targets at once,
but you cannot pass arguments to it via the command line.
Instead, set the `SPHINXOPTS` environment variable to pass arguments to `sphinx-build`.
