🤖 NexusAI Multi-Agent System 🚀

<p align="center">
  <img src="https://media.giphy.com/media/l0HlNaQ6gWfllcjDO/giphy.gif" width="700" />
</p><p align="center">
  ⚡ Ultra-Fast Multi-Agent AI System powered by Groq + LangGraph
</p>---

🧩 Tech Badges

<p align="center">
  <img src="https://img.shields.io/badge/AI-Multi--Agent-blueviolet"/>
  <img src="https://img.shields.io/badge/LLM-Groq-orange"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red"/>
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Flow-green"/>
  <img src="https://img.shields.io/badge/LangChain-Framework-yellow"/>
  <img src="https://img.shields.io/badge/FAISS-Memory-darkgreen"/>
  <img src="https://img.shields.io/badge/Generative%20AI-Agents-purple"/>
</p>---

🚀 Live Demo

👉 Try the App Here

🔗 Frontend (Streamlit):
https://multi-agent-system-ai.onrender.com/

---

📌 Project Overview

NexusAI is a Multi-Agent AI System designed to intelligently process user queries using a structured pipeline of AI agents.

Instead of relying on a single LLM call, this system breaks down tasks into multiple specialized agents, improving accuracy, reasoning, and response quality.

It combines speed + intelligence using Groq’s fast inference with a modular agent architecture.

---

✨ Key Features

⚡ Ultra-Fast Responses
Powered by Groq for low-latency inference

🧠 Multi-Agent Architecture
Planner → Worker → Reviewer pipeline

🚀 Smart Routing System
Simple queries bypass full pipeline for instant replies

📊 Context-Aware Memory
Stores and retrieves past interactions using FAISS

🔧 Tool Integration

- Web Search
- Calculator
- File Reader

🔁 Self-Improving Responses
Reviewer agent refines outputs

---

🛠️ Tech Stack

Technology| Purpose
🐍 Python| Core Programming
⚡ Groq API| Fast LLM Inference
🧠 LangGraph| Agent Workflow
🔗 LangChain| LLM Integration
🎨 Streamlit| Frontend UI
📦 FAISS| Vector Memory
🌐 Tavily| Web Search Tool

---

🏗️ Project Architecture

NexusAI
│
├── app.py              # Streamlit Frontend
├── agents.py           # Multi-Agent Logic
├── requirements.txt
├── .env                # Environment variables (NOT pushed)
└── README.md

---

⚙️ Installation Guide

1️⃣ Clone Repository

git clone https:https://github.com/hari9618/Multi-Agent_System
cd nexusai

---

2️⃣ Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

---

3️⃣ Install Dependencies

pip install -r requirements.txt

---

4️⃣ Setup Environment Variables

Create a ".env" file:

GROQ_API_KEY=your_api_key_here
AGENT_MODEL=gemma2-9b-it

---

5️⃣ Run the App

streamlit run app.py

---

🧠 How It Works

1️⃣ User sends a query
2️⃣ Smart Router decides execution path
3️⃣ Planner creates structured steps
4️⃣ Worker generates solution
5️⃣ Reviewer refines output
6️⃣ Final response displayed

---

📷 Application Preview
<img width="953" height="445" alt="Screenshot 2026-04-05 171741" src="https://github.com/user-attachments/assets/3ac92771-ef15-4ad3-b37c-10be33753870" />


---

📚 What I Learned

✔ Multi-Agent System Design
✔ LangGraph Workflow Engineering
✔ LLM Optimization for Speed
✔ Memory Integration with FAISS
✔ Tool-augmented AI Systems

---

🎯 Future Improvements

🔹 RAG-based knowledge integration
🔹 Advanced tool chaining
🔹 Real-time collaboration agents
🔹 Voice-based interaction
🔹 UI enhancements

---

👨‍💻 Author

Hari Krishna
AI Engineer | Multi-Agent Systems Builder

🔗 GitHub
https://github.com/hari9618

---

⭐ Support

If you like this project:

⭐ Star the repository
📢 Share with others

---

📢 Tags

AI • Multi-Agent • LangGraph • Groq • Streamlit • Python • Generative AI • LLM
