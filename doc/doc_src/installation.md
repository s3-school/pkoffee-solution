# Installation

Follow these steps to install the project and its dependencies:

1. **Clone the repository**:

```shell
git clone git@github.com:s3-school/draft-project.git s3draft && cd s3draft
```

2. **Create a workspace envorinment with pixi**
`s3draft` manages its dependencies with [`pixi`](https://pixi.sh/latest/). It provides tasks to run tests or run the dummy executable:

```shell
# install the development environment containing all dependencies, test and documentation tools
pixi install
# Run the tests, make coverage reports
pixi run test
# build the documentation
pixi run doc
# call the s3draft in the default environment of the workspace
pixi run s3draft
```
