FROM python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt \
    && python -c "import ctypes; ctypes.CDLL('libseccomp.so.2')"
COPY run_agent.py .
COPY ping_floyd/ ping_floyd/
ENTRYPOINT ["python", "/app/run_agent.py"]

FROM runtime AS bench
COPY bench/ bench/

FROM runtime AS contest
