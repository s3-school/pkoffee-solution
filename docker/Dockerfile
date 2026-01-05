# Example of a distroless image for running pkoffee in production
# More examples at https://github.com/pavelzw/pixi-docker-example

# To build the image locally, from the pkoffee-solution directory:
# docker build -f docker/pkoffee-base/Dockerfile -t pkoffee:local --build-arg INSTALL_ENV=prod .
# Can also use docker build options --no-cache --progress=plain for easier debugging

ARG INSTALL_ENV=prod

FROM ghcr.io/prefix-dev/pixi:0.61.0 AS build

# re-define arg from outer multistage to be able to use value inside stage
ARG INSTALL_ENV

# copy the entire project in the builder stage to make installation, excluding the .pixi directory
WORKDIR /app
COPY --exclude=.pixi/* . .
# install dependencies to `/app/.pixi/envs/$INSTALL_ENV`
# use `--locked` to ensure the lockfile is up to date with pixi.toml
# BUILD_EDITABLE required, because for now pixi install defaults to editable install
RUN BUILD_EDITABLE_PYTHON=false pixi install --locked -e "$INSTALL_ENV"


FROM ubuntu:24.04 AS production
ARG INSTALL_ENV
WORKDIR /app
# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/"$INSTALL_ENV" /app/.pixi/envs/"$INSTALL_ENV"

# Add the environment to PATH, similar to "pixi shell-hook -e $INSTALL_ENV"
ENV PATH=/app/.pixi/envs/"$INSTALL_ENV"/bin:$PATH
ENV CONDA_PREFIX=/app/.pixi/envs/"$INSTALL_ENV"
CMD [ "pkoffee" ]

# Example of using the local image
# docker run --mount type=bind,src=$(pwd)/doc/doc_src/example_data,dst=/example_data --mount type=bind,src=$(pwd)/results,destination=/analysis_results pkoffee:local pkoffee analyze --data-file /example_data/coffee_productivity.csv --output /analysis_results/analysis.png