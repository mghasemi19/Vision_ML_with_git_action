FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY pyproject.toml ./
RUN pip install --no-deps --no-cache-dir .
# Model is produced and quality-gated by CI, not trained in a request.
COPY models/model.joblib ./models/model.joblib
RUN useradd --system --uid 10001 appuser && chown -R appuser:appuser /service
USER appuser
EXPOSE 8000
CMD ["uvicorn", "vision_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
