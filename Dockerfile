FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
ENV PREDMAINT_STATE=/data
VOLUME /data
EXPOSE 8001
CMD ["uvicorn", "predmaint.serve:get_app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
