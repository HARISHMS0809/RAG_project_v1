
# RAG-Based Employee Onboarding Assistant [V1]

An AI-powered web application built with **Flask**, **FAISS**, and **Sentence-Transformers** that helps employees and new joiners **analyze and query onboarding documents** such as offer letters and joining letters — securely and intelligently.

---

##  Project Overview

This application uses **Retrieval-Augmented Generation (RAG)** to enable contextual question-answering on uploaded employee documents.  
It extracts text, indexes it into **FAISS** for vector-based semantic search, and uses transformer models to provide **context-aware responses**.

---

##  Features

-  **Document Upload** – Upload offer or joining letters (PDF/TXT/DOCX).  
-  **Text Extraction** – Automatically extract text from uploaded files.  
-  **Semantic Search** – Uses FAISS vector database for similarity-based retrieval.  
-  **Contextual Q&A** – Ask questions and get AI-generated answers from your document.  
-  **Secure Interface** – Local processing ensures privacy of employee data.
                          - user ID  = admin 
                          - Password = admin123 
-  **Flask Web App** – Easy-to-use and lightweight backend for deployment.

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Flask (Python) |
| **Embedding Model** | Sentence-Transformers |
| **Vector Store** | FAISS |
| **RAG Pipeline** | Retrieval-Augmented Generation |
| **Deployment** | Localhost / Cloud-ready (Render, AWS, etc.) |

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/HARISHMS0809/RAG_project_v1.git
   cd RAG_project_v1
