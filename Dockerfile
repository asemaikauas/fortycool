# Portable image, so this deploys the same way on Render, Railway, Fly, or any
# container host. Render can also build without it via render.yaml.

FROM python:3.12-slim AS base

# Never write .pyc, never buffer stdout: logs should appear in the platform's
# log viewer as they happen rather than when a buffer flushes.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall pandas.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install .

# Run as a non-root user. A web process has no reason to be able to write to
# its own application directory.
RUN useradd --create-home --uid 10001 fortycool \
 && mkdir -p /tmp/fortycool \
 && chown -R fortycool:fortycool /tmp/fortycool
USER fortycool

# Ephemeral by default. Mount a volume here to keep runs across restarts.
ENV FORTYCOOL_DB_PATH=/tmp/fortycool/runs.sqlite3 \
    FORTYCOOL_TRUSTED_PROXY_HOPS=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health', timeout=4).status==200 else 1)"

# 0.0.0.0, not 127.0.0.1: binding loopback inside a container makes the service
# unreachable from outside it. One worker, because the telemetry and job stores
# are per-process.
CMD ["sh", "-c", "exec python -m uvicorn fortycool_agents.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
