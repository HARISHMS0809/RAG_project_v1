from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from functools import wraps
from threading import Lock
import uuid
import json
import re
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['INDEX_FOLDER'] = 'index_store'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-production-via-env')

USERNAME = os.environ.get('ADMIN_USER', 'admin')
PASSWORD = os.environ.get('ADMIN_PASS', 'admin123')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_generator = pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
print("Models loaded successfully!")

dimension = 384
faiss_index = None
metadata_store = []
index_lock = Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(path):
    text = ""
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[PAGE {i+1}]\n{page_text}\n"
            except:
                continue
    return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def sentence_split(text):
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

def chunk_text_by_sentences(text, max_words=180, overlap_sentences=2):
    sentences = sentence_split(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = []
        words = 0
        j = i
        while j < len(sentences) and words + len(sentences[j].split()) <= max_words:
            chunk_sentences.append(sentences[j])
            words += len(sentences[j].split())
            j += 1
        if not chunk_sentences and j < len(sentences):
            chunk_sentences.append(sentences[j])
            j += 1
        if chunk_sentences:
            chunks.append((i, j, " ".join(chunk_sentences)))
        i = max(i + 1, j - overlap_sentences)
    return chunks

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embeddings / norms

def init_faiss_index(dimension):
    return faiss.IndexFlatIP(dimension)

def save_index_and_metadata(index, metadata, path_prefix):
    faiss.write_index(index, path_prefix + ".index")
    with open(path_prefix + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_index_and_metadata(path_prefix):
    if not os.path.exists(path_prefix + ".index"):
        return None, []
    index = faiss.read_index(path_prefix + ".index")
    with open(path_prefix + ".meta.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def mmr_rerank(query_embedding, candidate_embeddings, candidate_texts, top_k=3, lambda_param=0.7):
    if candidate_embeddings.shape[0] == 0:
        return []
    query_sim = (candidate_embeddings @ query_embedding).flatten()
    selected_indices = [int(np.argmax(query_sim))]
    
    while len(selected_indices) < min(top_k, candidate_embeddings.shape[0]):
        remaining = [i for i in range(candidate_embeddings.shape[0]) if i not in selected_indices]
        mmr_scores = []
        for i in remaining:
            sim_to_query = query_sim[i]
            sim_to_selected = max([float(candidate_embeddings[i] @ candidate_embeddings[j]) 
                                   for j in selected_indices])
            mmr_score = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((mmr_score, i))
        mmr_scores.sort(reverse=True)
        selected_indices.append(mmr_scores[0][1])
    return selected_indices

def add_chunks_to_index(chunks_with_meta):
    global faiss_index, metadata_store
    with index_lock:
        texts = [c['text'] for c in chunks_with_meta]
        embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=False, 
                                           convert_to_numpy=True).astype('float32')
        embeddings = normalize_embeddings(embeddings)
        if faiss_index is None:
            faiss_index = init_faiss_index(dimension)
        faiss_index.add(embeddings)
        metadata_store.extend(chunks_with_meta)

def retrieve(query, top_k=5, mmr_k=3):
    if faiss_index is None or len(metadata_store) == 0:
        return []
    
    q_emb = embedding_model.encode([query], convert_to_numpy=True)[0].astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    
    search_k = min(max(top_k, mmr_k) + 10, faiss_index.ntotal)
    D, I = faiss_index.search(np.expand_dims(q_emb, axis=0), search_k)
    indices = [int(i) for i in I[0] if i != -1]
    
    if not indices:
        return []
    
    candidate_meta = [metadata_store[idx] for idx in indices]
    candidate_texts = [m['text'] for m in candidate_meta]
    candidate_embeddings = embedding_model.encode(candidate_texts, convert_to_numpy=True).astype('float32')
    candidate_embeddings = normalize_embeddings(candidate_embeddings)
    
    selected_idxs = mmr_rerank(q_emb, candidate_embeddings, candidate_texts, top_k=mmr_k, lambda_param=0.7)
    return [candidate_meta[i] for i in selected_idxs]

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    return redirect(url_for('main') if 'logged_in' in session else url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == USERNAME and request.form.get('password') == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('main'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/main')
@login_required
def main():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            raw_text = extract_text_from_pdf(path)
        elif ext == 'docx':
            raw_text = extract_text_from_docx(path)
        else:
            raw_text = extract_text_from_txt(path)
        
        chunk_tuples = chunk_text_by_sentences(raw_text, max_words=180, overlap_sentences=2)
        chunks_meta = [{
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "source": filename,
            "range": [start_idx, end_idx],
            "uploaded_at": datetime.now().isoformat()
        } for start_idx, end_idx, chunk_text in chunk_tuples]
        
        add_chunks_to_index(chunks_meta)
        save_index_and_metadata(faiss_index, metadata_store, 
                               os.path.join(app.config['INDEX_FOLDER'], 'faiss_rag'))
        os.remove(path)
        
        return jsonify({'success': True, 'chunks_indexed': len(chunks_meta), 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
@login_required
def query_document():
    data = request.get_json(force=True)
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    if faiss_index is None or len(metadata_store) == 0:
        return jsonify({'error': 'No documents indexed. Please upload documents first.'}), 400
    
    try:
        retrieved = retrieve(query, top_k=10, mmr_k=4)
        combined_context = "\n\n".join([f"[{r['source']}] {r['text']}" for r in retrieved])
        
        prompt = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nProvide a clear, concise answer based on the context above:"
        answer = qa_generator(prompt, max_length=256, truncation=True, do_sample=False)[0]['generated_text']
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': list(set([r['source'] for r in retrieved])),
            'retrieved': [{
                'source': r['source'],
                'text': r['text'][:300] + '...' if len(r['text']) > 300 else r['text']
            } for r in retrieved]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
@login_required
def get_stats():
    with index_lock:
        total_chunks = len(metadata_store)
        sources = list(set([m['source'] for m in metadata_store])) if metadata_store else []
        return jsonify({
            'total_chunks': total_chunks,
            'total_documents': len(sources),
            'documents': sources
        })

@app.route('/reset', methods=['POST'])
@login_required
def reset_index():
    global faiss_index, metadata_store
    with index_lock:
        faiss_index = None
        metadata_store = []
        base = os.path.join(app.config['INDEX_FOLDER'], 'faiss_rag')
        for ext in ['.index', '.meta.json']:
            p = base + ext
            if os.path.exists(p):
                os.remove(p)
    return jsonify({'success': True})

def try_load_index():
    global faiss_index, metadata_store
    base = os.path.join(app.config['INDEX_FOLDER'], 'faiss_rag')
    idx, meta = load_index_and_metadata(base)
    if idx is not None and meta:
        faiss_index = idx
        metadata_store = meta
        print(f"Loaded {len(metadata_store)} chunks from persistent storage")

if __name__ == '__main__':
    # Load index at startup
    try_load_index()
    app.run(debug=False, host='0.0.0.0', port=5000)