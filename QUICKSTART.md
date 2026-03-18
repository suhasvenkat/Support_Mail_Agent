# Quick Start Guide

Get Support Mail Agent running in 5 minutes.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment

Create `.env` file:
```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 3. Start FastAPI Backend

In Terminal 1:
```bash
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 4. Start Streamlit Frontend

In Terminal 2:
```bash
streamlit run streamlit_app.py
```

Streamlit will automatically open in your browser at `http://localhost:8501`

## 5. Test It!

1. Enter a customer email:
   - **From:** customer@example.com
   - **Subject:** I need help with billing
   - **Body:** I was charged twice for my subscription

2. Click "🚀 Process Email"

3. See results:
   - Intent classification
   - Confidence score
   - Knowledge base results
   - AI-generated response
   - Escalation status

## That's it! 🎉

Your Support Mail Agent is now running locally.

### Next Steps

- **Add knowledge base docs** to `knowledge_base/docs/` folder
- **Test different email types** (billing, technical, complaints)
- **Review history** in the "History" tab
- **Deploy to production** when ready (see [DEPLOYMENT.md](DEPLOYMENT.md))

### Troubleshooting

**FastAPI won't start:**
- Make sure port 8000 is free: `lsof -i :8000`
- Check ANTHROPIC_API_KEY is set: `echo $ANTHROPIC_API_KEY`

**Streamlit says "Cannot connect to API":**
- Make sure FastAPI is running (Terminal 1)
- Check http://localhost:8000/health in browser

**Need more help?**
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup
- Review [README.md](README.md) for project overview
