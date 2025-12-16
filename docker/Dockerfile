# Example of a distroless image for running pkoffee in production
# More examples at https://github.com/pavelzw/pixi-docker-example

# To build the image locally, from the pkoffee-solution directory:
# docker build -f docker/Dockerfile -t pkoffee:local .
# Can also use docker build options --no-cache --progress=plain for easier debugging

FROM ghcr.io/prefix-dev/pixi:0.61.0 AS build

# copy the entire project in the builder stage to make installation, excluding the .pixi directory
WORKDIR /app
COPY --exclude=.pixi/* . .
# install dependencies to `/app/.pixi/envs/prod`
# use `--locked` to ensure the lockfile is up to date with pixi.toml
# BUILD_EDITABLE required, because for now pixi install defaults to editable install
RUN BUILD_EDITABLE_PYTHON=false pixi install --locked -e prod


FROM ubuntu:24.04 AS production
WORKDIR /app
# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod

# Add the environment to PATH, similar to "pixi shell-hook -e prod"
ENV PATH=/app/.pixi/envs/prod/bin:$PATH
ENV CONDA_PREFIX=/app/.pixi/envs/prod
CMD [ "pkoffee" ]

# Example of using the local image
# docker run --mount type=bind,src=$(pwd)/doc/doc_src/example_data,dst=/example_data --mount type=bind,src=$(pwd)/results,destination=/analysis_results pkoffee:local pkoffee analyze --data-file /example_data/coffee_productivity.csv --output /analysis_results/analysis.png