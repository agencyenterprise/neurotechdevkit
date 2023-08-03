FROM python:3.10.0 as build

WORKDIR /ndk

COPY pyproject.toml ./pyproject.toml
COPY poetry.lock ./poetry.lock
COPY README.md ./README.md
COPY Makefile ./Makefile
COPY src ./src
COPY tests ./tests
COPY run_jupyter_server.sh ./run_jupyter_server.sh
RUN ["chmod", "+x", "./run_jupyter_server.sh"]

# Use pipx to install poetry into its own venv
# in case poetry's dependencies conflict with ndk's dependencies
RUN python3 -m pip install --user "pip>=19.0" && \
    python3 -m pip install --user pipx
# ENV line is equivalent to: RUN python3 -m pipx ensurepath
ENV PATH=/root/.local/bin:$PATH
RUN pipx install poetry

# Install ndk's dependencies to venv
RUN python3 -m venv ./venv
ENV VIRTUAL_ENV=./venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install --all-extras

FROM python:3.10.0-slim

COPY --from=build /ndk /ndk
WORKDIR /ndk

RUN ./venv/bin/pip install --upgrade pip

RUN apt-get update && \
    apt-get install -y g++ jq make unzip wget ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ./venv/bin/pip install jupyter

ENV PATH "/ndk/venv/bin:$PATH"
ENV DEVITO_ARCH "gcc"

LABEL org.opencontainers.image.source="https://github.com/agencyenterprise/neurotechdevkit"

RUN ./venv/bin/pip install .

WORKDIR /ndk/notebooks
RUN wget "https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip" -O temp.zip && unzip temp.zip && rm temp.zip

EXPOSE 8888
CMD ["/ndk/run_jupyter_server.sh"]
