# SafePause service, s390x
# based on the LinuxONE lab image (tensorflow + zdnn plugin, xgboost, llama.cpp, granite)
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SAFEPAUSE_HOME=/app \
    SAFEPAUSE_ARTIFACTS=/app/artifacts

WORKDIR /app
COPY safepause /app/safepause
COPY scripts /app/scripts

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health', timeout=4)" || exit 1

ENTRYPOINT []
CMD ["python", "-m", "safepause.server", "--host", "0.0.0.0", "--port", "8080"]
