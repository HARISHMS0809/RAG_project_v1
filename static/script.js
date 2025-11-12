// Global state
let isUploading = false;
let isQuerying = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    const queryForm = document.getElementById('queryForm');
    
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    if (queryForm) {
        queryForm.addEventListener('submit', handleQuery);
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        updateStatsDisplay(data);
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Update stats display
function updateStatsDisplay(data) {
    const statsText = document.getElementById('statsText');
    if (!statsText) return;
    
    if (data.total_documents === 0) {
        statsText.textContent = 'No documents';
    } else {
        statsText.textContent = `${data.total_documents} document${data.total_documents > 1 ? 's' : ''} • ${data.total_chunks} chunks`;
    }
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const fileNameDisplay = document.getElementById('fileName');
    fileNameDisplay.textContent = file.name;
    
    uploadFile(file);
}

// Upload file
async function uploadFile(file) {
    if (isUploading) return;
    
    isUploading = true;
    const progressDiv = document.getElementById('uploadProgress');
    const fileNameDisplay = document.getElementById('fileName');
    
    try {
        progressDiv.style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccessToast(`Successfully indexed ${data.chunks_indexed} chunks from ${data.filename}`);
            fileNameDisplay.textContent = '';
            document.getElementById('fileInput').value = '';
            loadStats();
            
            // Remove welcome message if exists
            const welcomeMsg = document.querySelector('.welcome-message');
            if (welcomeMsg) {
                welcomeMsg.remove();
            }
        } else {
            showErrorToast(data.error || 'Upload failed');
        }
    } catch (error) {
        showErrorToast('Network error during upload');
        console.error(error);
    } finally {
        progressDiv.style.display = 'none';
        isUploading = false;
    }
}

// Handle query submission
async function handleQuery(e) {
    e.preventDefault();
    
    if (isQuerying) return;
    
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) return;
    
    isQuerying = true;
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    
    // Add user message
    addMessage('user', query);
    input.value = '';
    
    // Add loading indicator
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingMessage(loadingId);
        
        if (data.success) {
            addMessage('assistant', data.answer, data.sources, data.retrieved);
        } else {
            addMessage('assistant', `Error: ${data.error}`);
            showErrorToast(data.error);
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage('assistant', 'Network error. Please try again.');
        showErrorToast('Network error during query');
        console.error(error);
    } finally {
        isQuerying = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// Add message to chat
function addMessage(role, content, sources = null, retrieved = null) {
    const chatHistory = document.getElementById('chatHistory');
    
    // Remove welcome message if exists
    const welcomeMsg = chatHistory.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    // Add label
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = role === 'user' ? 'You' : 'Assistant';
    bubbleDiv.appendChild(label);
    
    // Add content
    const contentP = document.createElement('p');
    contentP.textContent = content;
    bubbleDiv.appendChild(contentP);
    
    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.textContent = 'Sources:';
        sourcesDiv.appendChild(sourcesTitle);
        
        sources.forEach(source => {
            const tag = document.createElement('span');
            tag.className = 'source-tag';
            tag.textContent = source;
            sourcesDiv.appendChild(tag);
        });
        
        bubbleDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(bubbleDiv);
    chatHistory.appendChild(messageDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const chatHistory = document.getElementById('chatHistory');
    const loadingId = 'loading-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message message-assistant';
    messageDiv.id = loadingId;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message-loading';
    loadingDiv.innerHTML = `
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <span>Searching documents...</span>
    `;
    
    messageDiv.appendChild(loadingDiv);
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingMsg = document.getElementById(loadingId);
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

// Reset index
async function resetIndex() {
    if (!confirm('Are you sure you want to reset all documents? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch('/reset', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccessToast('All documents have been reset');
            
            // Clear chat history
            const chatHistory = document.getElementById('chatHistory');
            chatHistory.innerHTML = `
                <div class="welcome-message">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                    <h3>Welcome!</h3>
                    <p>Upload your onboarding documents and start asking questions.</p>
                </div>
            `;
            
            // Update stats
            loadStats();
        }
    } catch (error) {
        showErrorToast('Failed to reset documents');
        console.error(error);
    }
}

// Toast notifications
function showSuccessToast(message) {
    showToast(message, 'success');
}

function showErrorToast(message) {
    showToast(message, 'error');
}

function showToast(message, type = 'success') {
    // Remove existing toasts
    const existing = document.querySelectorAll('.toast');
    existing.forEach(t => t.remove());
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : '#ef4444'};
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideIn 0.3s ease;
        max-width: 400px;
        font-size: 14px;
        font-weight: 500;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);