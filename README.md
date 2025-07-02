# 🧠 Social Support Application Prototype (Agentic AI + LangGraph)

A modular, containerized GenAI-powered application that automates financial aid eligibility using documents, LLM reasoning, and ML scoring — orchestrated with LangGraph and monitored with LangSmith.

---

## 🚀 Features

- 🔍 **Document Parsing (PDF/Excel)** using LlamaIndex,openpyxl & pandas
- 🔁 **Data Reconciliation Agent** to compare user form vs. documents
- 🔐 **Validation Agent** for Gov & Bank verification
- 🧠 **ML Agent (XGBoost + SHAP)** for eligibility prediction
- 🤖 **Ollama/LLM agent** for personalized AI decision messages
- 🎓 **Career Recommendation Agent** (optional resume parsing (PyMuPDF) and career recommendation)
- 📊 **LangSmith Tracing** for workflow insights
- 🌐 **Streamlit UI** with real-time status
- 🐳 Fully **Dockerized** for local/offline LLM use
- ☸️ Optional **Kubernetes support** for scalable agent deployment

---

## 🧩 Agentic AI Workflow (LangGraph)

LangGraph orchestrates the flow between these agent nodes:

```
extract_documents → reconcile_data → run_validation → 
evaluate_financial_assistance → generate_recommendations
```

Each node is decorated with `@traceable` (LangSmith) and emits events for observability.

📁 Agent functions used (see `workflow/workflow.py`):
- `extract_documents_node`: parses Emirates ID & bank statement
- `reconcile_data_node`: checks field mismatches between document submitted and form submitted
- `run_validation_node`: validates with mock external services
- `evaluate_financial_assistance_node`: uses ML + LLM
- `generate_recommendations_node`: resume-based job tips

---

## 📁 Project Structure

```
social-support-prototype/
│
├── agents/                 # Modular agents (parser, reconcilliation, chatbot, validation, recommender)
├── workflow/               # LangGraph orchestrator logic
├── ui/                     # Streamlit UI with agent output history
├── utils/                  # XGBoost model, logger, status tracker
├── Dockerfile              # Ollama + Streamlit
├── start.sh                # Launch Ollama + app inside Docker
├── requirements.txt        
├── .env                    # LangSmith, OpenAI, etc.
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.12+ (for local run)
- [Ollama](https://ollama.com/) installed (optional)
- [Docker](https://docs.docker.com/get-docker/)
- (Optional) LangSmith account

---

## 🚀 Quick Start (Docker)

### 🔧 1. Clone the repo
```bash
git clone https://github.com/CodeWithAnkit1/socialsupportprototype.git
cd socialsupportprototype
```

### 📄 2. Create `.env`

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=Social_Support_App
OPENAI_API_KEY=your-key (optional)
```

### 🛠️ 3. Build and Run Docker

```bash
docker build -t social-support-app .
docker run -p 8501:8501 social-support-app
```

Access: [http://localhost:8501](http://localhost:8501)

---

## 🧠 LangSmith Integration

We use LangSmith for:
- ✍️ Tracing each agent call and LLM tool
- 🧪 Visual debugging
- 📊 Experiment tracking

✅ Enabled automatically via `.env`  
🌐 Visit [smith.langchain.com](https://smith.langchain.com) to view traces

---

## ☸️ Kubernetes Deployment (Advanced)

### Why?
Split agents into **microservices** for:
- Scalability (e.g. high-load for LLM or parsing agents)
- Fault isolation (e.g. if validator fails, rest still work)
- Reusability across projects

### Suggested Deployment Model

| Agent                   | Microservice Pod             |
|-------------------------|------------------------------|
| Document Parser         | `doc-parser-service`         |
| Validator (Gov/Bank)    | `validation-service`         |
| Reconciliation Agent    | `reconciliation-service`     |
| XGBoost Model Agent     | `eligibility-service`        |
| Recommendation Agent    | `recommendation-service`     |
| LangGraph Orchestrator  | `orchestrator-service`       |
| Streamlit UI            | `frontend-service`           |

Each service can be deployed as:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: doc-parser-service
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: parser
        image: your-org/doc-parser:latest
```

Use `nginx` or `traefik` as ingress for routing.

---

## 🧪 Sample Use Case

1. User uploads Emirates ID (PDF) and bank statement (Excel)
2. App extracts and reconciles form data vs docs
3. Validators check with mock APIs
4. ML model predicts eligibility
5. LLM generates explanation and job advice
6. All steps traced via LangSmith

---

## 📝 Customization

- 🔍 Add your own LLM or agents in `agents/`
- 🧠 Use `@traceable` on new tools for LangSmith visibility
- 🧱 Extend `workflow.py` with your own LangGraph logic

---

## 👤 Author

**Ankit Srivastava**  
Senior Data Scientist | BFSI & GenAI Specialist  
[LinkedIn](https://www.linkedin.com/in/ankit-srivastava-machinelearning/)

---

## 📜 License

MIT License

---

🧠 Powered by: LangGraph • LangChain • Streamlit • Ollama • LlamaIndex • XGBoost • LangSmith