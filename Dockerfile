FROM python:3.10.0 as build

RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false
COPY . ./

RUN /venv/bin/poetry install && \
    /venv/bin/pip install git+https://github.com/trustimaging/stride

FROM python:3.10.0-slim

RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip

COPY --from=build /venv /venv

RUN /venv/bin/pip install jupyter

COPY . /app
RUN rm -rf /app/.git

ENV PATH "/venv/bin:$PATH"

LABEL org.opencontainers.image.source="https://github.com/agencyenterprise/neurotechdevkit"

RUN apt-get update && apt-get install -y gcc gcc+ make

WORKDIR /app
RUN /venv/bin/pip install .

EXPOSE 8888
CMD ["/venv/bin/jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
