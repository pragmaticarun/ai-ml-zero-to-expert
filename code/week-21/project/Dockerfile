# UNTESTED: there is no Docker on the course box, so this file has never been built or run.
# Build context: code/week-21 (so _mlops.py is not needed at serve time; only project/ is copied).
FROM python:3.13-slim
RUN useradd --create-home app
WORKDIR /srv
COPY project/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY project/service.py project/artifacts/ ./
# artifacts are copied flat; point the service at them
ENV ARTIFACTS=/srv
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
