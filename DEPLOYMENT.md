# Deployment Guide - Support Mail Agent

This guide covers running the Support Mail Agent locally and deploying to production.

## Architecture Overview

The project consists of two main components:

1. **FastAPI Backend** (`main.py`) - Processes emails through LangGraph workflow
2. **Streamlit Frontend** (`streamlit_app.py`) - Interactive UI for testing and demos

```
┌─────────────────────┐
│  Streamlit Frontend │
│  (streamlit_app.py) │
└──────────┬──────────┘
           │ HTTP API calls
           ▼
┌──────────────────────────────────────┐
│      FastAPI Backend (main.py)       │
│  ┌──────────────────────────────────┐│
│  │     LangGraph Workflow           ││
│  │  - Classify Intent               ││
│  │  - Search Knowledge Base         ││
│  │  - Generate Response             ││
│  │  - Escalate (if needed)          ││
│  │  - Schedule Follow-ups           ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
           │
           ▼
    ┌────────────────┐
    │ Claude API     │
    │ Vector DB      │
    │ Knowledge Base │
    └────────────────┘
```

## Local Development

### Prerequisites

- Python 3.9+
- pip or conda
- Anthropic API key (set in `.env`)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=sk-ant-...
# Other configuration as needed
```

### 3. Start FastAPI Backend

```bash
python main.py
```

The API will run at `http://localhost:8000`

Test the API health:
```bash
curl http://localhost:8000/health
```

### 4. In a New Terminal, Start Streamlit Frontend

```bash
streamlit run streamlit_app.py
```

Streamlit will open at `http://localhost:8501`

### 5. Test the Application

1. Go to http://localhost:8501
2. Paste a sample support email
3. Click "Process Email"
4. View results and history

## Production Deployment

### Option 1: Deploy on Streamlit Cloud (Recommended)

**Pros:**
- Free tier available
- No server management
- Automatic HTTPS
- Easy CI/CD integration

**Steps:**

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit interface"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file to `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to your app settings
   - Add secret:
     ```
     API_BASE_URL = https://your-deployed-api.com
     ```

4. **Deploy FastAPI Backend** (see options below)

### Option 2: Deploy FastAPI on Railway/Render

**Railway (Recommended for simplicity):**

1. Create `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. Push to GitHub and deploy:
   - Go to [railway.app](https://railway.app)
   - Connect GitHub repo
   - Set start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - Add environment variables (ANTHROPIC_API_KEY, etc.)
   - Deploy

**Render.com Alternative:**

1. Go to [render.com](https://render.com)
2. Create new "Web Service"
3. Connect GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Add environment variables
7. Deploy

### Option 3: Deploy to AWS (Production-Grade)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    command: uvicorn main:app --host 0.0.0.0 --port 8000
```

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Deploy to AWS ECS/Fargate or EC2.

## Configuration

### Streamlit Secrets (.streamlit/secrets.toml)

Create this file for local development:
```toml
API_BASE_URL = "http://localhost:8000"
```

For production, set this in Streamlit Cloud dashboard or environment variable.

### Environment Variables

Required in FastAPI deployment:
- `ANTHROPIC_API_KEY` - Your Claude API key
- `ENVIRONMENT` - `development` or `production`

Optional:
- `KNOWLEDGE_BASE_PATH` - Path to knowledge base docs
- `LOG_LEVEL` - Logging level (default: `info`)

## Monitoring & Logging

### Local Development

Logs print to console automatically.

### Production

Configure application logging:
- Streamlit logs go to Streamlit Cloud dashboard
- FastAPI logs can be captured via:
  - CloudWatch (AWS)
  - Datadog
  - New Relic
  - Custom logging to external service

## Troubleshooting

### Streamlit can't connect to API

**Error:** "Cannot connect to API at http://localhost:8000"

**Solution:**
- Make sure FastAPI is running: `python main.py`
- Check FastAPI is listening: `curl http://localhost:8000/health`
- Verify `API_BASE_URL` in `.streamlit/secrets.toml`

### API returns 500 error

**Solution:**
- Check FastAPI logs for error details
- Verify `ANTHROPIC_API_KEY` is set and valid
- Ensure knowledge base is loaded

### Streamlit app won't start

**Error:** "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
# or
pip install -r requirements.txt
```

## Scaling Considerations

For production with high email volume:

1. **Database** - Replace in-memory storage in `emails.py` with PostgreSQL/MongoDB
2. **Queue** - Use Celery + Redis for async processing
3. **Caching** - Add Redis for knowledge base caching
4. **Load Balancing** - Use nginx/HAProxy with multiple API instances
5. **Monitoring** - Add Sentry for error tracking

## Example Production Setup

```
┌──────────────────────────────────┐
│  Streamlit Cloud                 │
│  (streamlit_app.py)              │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  CloudFront CDN                  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  AWS ALB (Load Balancer)         │
└────────────┬─────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐       ┌─────────┐
│ ECS     │       │ ECS     │
│ Task 1  │       │ Task 2  │
│(FastAPI)│       │(FastAPI)│
└────┬────┘       └────┬────┘
     │                 │
     └────────┬────────┘
              ▼
    ┌──────────────────┐
    │ RDS (PostgreSQL) │
    │ ElastiCache      │
    │ (Redis)          │
    └──────────────────┘
```

## Next Steps

1. Test the Streamlit interface locally
2. Deploy FastAPI backend to production
3. Update `.streamlit/secrets.toml` with production API URL
4. Deploy Streamlit app to Streamlit Cloud
5. Set up monitoring and logging
6. Configure CI/CD for automatic deploys

## Resources

- [Streamlit Deployment Documentation](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Claude API Documentation](https://docs.anthropic.com/)
