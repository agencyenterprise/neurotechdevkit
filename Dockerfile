FROM python:3.11.0 as build

# Install system dependencies for HDF5
RUN apt-get update && apt-get install -y \
    libhdf5-dev python3-h5py \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ndk

COPY pyproject.toml ./pyproject.toml
COPY poetry.lock ./poetry.lock
COPY README.md ./README.md
COPY Makefile ./Makefile
COPY src ./src
COPY tests ./tests
COPY run_jupyter_server.sh ./run_jupyter_server.sh
RUN ["chmod", "+x", "./run_jupyter_server.sh"]

RUN python3 -m venv ./venv && \
    ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false
RUN ./venv/bin/poetry install && \
    ./venv/bin/pip install git+https://github.com/trustimaging/stride@2520c59

FROM python:3.11.0-slim

COPY --from=build /ndk /ndk
WORKDIR /ndk

RUN ./venv/bin/pip install --upgrade pip

RUN apt-get update && \
    apt-get install -y g++ jq make unzip wget ffmpeg libhdf5-dev python3-h5py  && \
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
