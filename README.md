# SupportMailAgent 🤖

An AI-powered customer support email agent that automates email triage, classification, knowledge base retrieval, and intelligent response generation with human escalation for complex cases.

**Built with:** LangGraph • LangChain • FastAPI • FAISS • MockLLM (works without API credits!)

---

## ✨ Features

✅ **Intent Classification** — Automatically categorizes emails (billing, technical, general, complaint, urgent)
✅ **Semantic Search** — Finds relevant knowledge base articles using FAISS vector search
✅ **AI Response Generation** — Drafts professional customer responses with context
✅ **Smart Escalation** — Routes complex issues to human agents based on confidence threshold
✅ **Web Dashboard** — Beautiful UI to compose test emails and view results
✅ **Inbox Viewer** — See all processed emails with workflow details
✅ **Mock Mode** — Works without OpenAI API credits (great for students!)

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd SupportMailAgent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env if you have OpenAI API key, otherwise leave MOCK_MODE=true
```

### 3. Load Knowledge Base (Optional)

```bash
python cli_kb_manager.py load
```

### 4. Start Server

```bash
bash SETUP_AND_RUN.sh
# Or manually:
uvicorn main:app --reload
```

### 5. Open in Browser

- **Dashboard (Compose):** http://localhost:8000
- **Inbox (View Results):** http://localhost:8000/inbox.html
- **API Docs:** http://localhost:8000/docs

---

## 📊 How It Works

```
EMAIL IN
   ↓
[1] CLASSIFIER → Determine intent & category
   ↓
[2] KB RETRIEVER → Search knowledge base for relevant docs
   ↓
[3] RESPONDER → Generate AI response with KB context
   ↓
[4] ESCALATOR → Check confidence, decide if human needed
   ↓
EMAIL OUT (auto-reply or escalation)
```

### Escalation Logic

An email is **escalated to human** if ANY of these:
- Confidence score < 70%
- Intent is `urgent` or `complaint`
- No knowledge base results found
- Processing errors occur

---

## 🎯 Mock Mode (No API Costs)

Your project works perfectly **without** OpenAI API credits:

| Feature | Real API | Mock Mode |
|---------|----------|-----------|
| Intent Classification | ✅ OpenAI | ✅ Keyword-based |
| Vector Embeddings | ✅ OpenAI | ✅ SHA256 deterministic |
| LLM Responses | ✅ OpenAI | ✅ Rule-based realistic |
| FAISS Search | ✅ Works | ✅ Works identically |
| **Cost** | 💰 | **FREE** |

**To use real OpenAI API later:**
```bash
# Edit .env
MOCK_MODE=false
OPENAI_API_KEY=sk-...
```

No code changes needed — auto-switches!

---

## 📁 Project Structure

```
SupportMailAgent/
├── main.py                      # FastAPI app entry point
├── cli_kb_manager.py            # CLI for loading knowledge base
├── requirements.txt             # Dependencies
├── .env.example                 # Config template
│
├── src/
│   ├── api/routes/emails.py    # POST /emails/process, GET /emails
│   ├── graph/
│   │   ├── state.py            # Workflow state definition
│   │   └── workflow.py         # LangGraph orchestration
│   ├── nodes/                  # Workflow nodes
│   │   ├── classifier.py       # Classify intent
│   │   ├── kb_retriever.py     # Search knowledge base
│   │   ├── responder.py        # Draft response
│   │   ├── escalator.py        # Escalation logic
│   │   └── followup.py         # Schedule follow-ups
│   ├── services/               # Business logic
│   │   ├── mock_llm.py         # MockLLM (works without API)
│   │   ├── mock_embeddings.py  # Mock vector embeddings
│   │   ├── faiss_store.py      # FAISS vector store
│   │   ├── email_service.py    # Email operations
│   │   └── followup_service.py # Follow-up scheduling
│   ├── prompts/                # LLM prompt templates
│   ├── schemas/                # Pydantic data models
│   ├── core/
│   │   ├── config.py           # Settings from .env
│   │   └── llm.py              # LLM factory (real or mock)
│   └── utils/                  # Utilities
│
├── static/                     # Web UI
│   ├── index.html              # Dashboard
│   ├── inbox.html              # Inbox viewer
│   ├── js/app.js               # Frontend logic
│   └── css/style.css           # Styling
│
├── knowledge_base/
│   ├── docs/                   # FAQ/documentation (.txt files)
│   └── loader.py               # Document loader
│
├── tests/                      # Unit tests
└── data/                       # FAISS index & metadata
```

---

## 💻 API Reference

### Process Email

```bash
POST /emails/process
Content-Type: application/json

{
  "sender": "customer@example.com",
  "subject": "I was charged twice",
  "body": "My credit card was charged twice for my subscription..."
}
```

**Response:**
```json
{
  "email_id": "email_abc123",
  "recipient": "customer@example.com",
  "subject": "Re: I was charged twice",
  "body": "Thank you for reaching out...",
  "escalated": false
}
```

### List All Emails

```bash
GET /emails
```

### Get Email Details

```bash
GET /emails/{email_id}
```

Returns full workflow execution details including intent, confidence, KB results, and AI response.

---

## 🧪 Testing

### From Dashboard

1. Open http://localhost:8000
2. Choose a template (Billing, Technical, Password, Complaint)
3. Click "Send Email"
4. Check Inbox at http://localhost:8000/inbox.html

### From Terminal

```bash
curl -X POST http://localhost:8000/emails/process \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "test@example.com",
    "subject": "My app keeps crashing",
    "body": "Every time I try to log in, I get an error message."
  }'
```

### Expected Results

| Email Type | Intent | Confidence | Action |
|-----------|--------|-----------|--------|
| "I was charged twice" | `billing` | 85% | ✅ Auto-Reply |
| "App keeps crashing" | `technical` | 85% | ✅ Auto-Reply |
| "Your service is terrible" | `complaint` | 85% | 🚨 Escalated |
| "I forgot my password" | `general` | 85% | ✅ Auto-Reply |

---

## 📚 Knowledge Base Setup

Add FAQ/documentation as `.txt` files in `knowledge_base/docs/`:

```bash
cat > knowledge_base/docs/billing_faq.txt << 'EOF'
Q: Why was I charged twice?
A: Double charges typically occur during subscription renewal.
Our team reviews all charges within 24 hours.
Reply with your transaction ID and we'll investigate.

Q: How do I upgrade my plan?
A: Go to Settings > Billing > Plan and select your tier.
Changes take effect immediately.
EOF
```

Then load:
```bash
python cli_kb_manager.py load
```

---

## 🔧 Configuration

Edit `.env`:

```env
# OpenAI API (optional - mock mode works without this)
OPENAI_API_KEY=sk-your-key-here

# Application
APP_ENV=development
LOG_LEVEL=INFO

# Mock mode (set to false when you have API credits)
MOCK_MODE=true

# Data storage
CHROMA_PATH=./data/chroma_db
```

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph |
| **LLM Framework** | LangChain |
| **API** | FastAPI + Uvicorn |
| **Vector Search** | FAISS |
| **Data Validation** | Pydantic |
| **Frontend** | Vanilla JS + CSS Grid |

---

## 🚀 Deployment

### Local Development
```bash
uvicorn main:app --reload
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```bash
docker build -t supportmailagent .
docker run -p 8000:8000 supportmailagent
```

---

## 📝 License

MIT

---

## 🤝 Contributing

Pull requests welcome! Please ensure tests pass:

```bash
pytest tests/
```

---

## 📖 Learn More

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Docs](https://python.langchain.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

---

**Built with ❤️ for efficient customer support automation**
