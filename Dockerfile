FROM mambaorg/micromamba:1.3.1

USER root

RUN set -ex \
    && apt-get -y update && apt-get install -y \
    curl \
    git \
    bash-completion \
    r-base

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yaml /tmp/environment.yaml

RUN micromamba install -y -n base -f /tmp/environment.yaml && \
    micromamba clean --all --yes
