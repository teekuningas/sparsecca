FROM mambaorg/micromamba:1.3.1

USER root

RUN set -ex \
    && apt-get -y update && apt-get install -y \
    curl \
    git \
    bash-completion \
    r-base \
    gcc \
    libglpk-dev \
    coinor-libipopt-dev \
    glpk-utils


USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Install black & sparsecca
RUN source /usr/local/bin/_activate_current_env.sh && \
    pip install black && \
    pip install .
