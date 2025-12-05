FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "app.generate"]
CMD ["--seed", "42", "--companies", "10", "--industries", "SaaS,Healthcare,Financial Services,Manufacturing", "--outdir", "/app/output"]
