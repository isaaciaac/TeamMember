ARG PYTHON_BASE_IMAGE=python:3.11-slim
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Note:
# Some networks block Debian apt repositories, which breaks `apt-get update` during image build.
# We intentionally avoid apt-get here; pip/httpx rely on certifi and do not require OS-level ca-certificates.

ARG PIP_INDEX_URL=
ARG PIP_TRUSTED_HOST=

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
  && if [ -n "${PIP_INDEX_URL}" ]; then pip config set global.index-url "${PIP_INDEX_URL}"; fi \
  && if [ -n "${PIP_TRUSTED_HOST}" ]; then pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; fi \
  && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

CMD ["python", "-m", "app.server"]
