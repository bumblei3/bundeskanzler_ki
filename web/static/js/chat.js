/**
 * Chat-Manager für Bundeskanzler KI Web Interface
 * Verwaltet Chat-Nachrichten, KI-Interaktion und Chat-Verlauf
 */

class ChatManager {
    constructor() {
        this.messages = [];
        this.currentModel = 'bundeskanzler';
        this.isTyping = false;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadChatHistory();
        this.setupAutoResize();
    }

    setupEventListeners() {
        // Send message
        const sendBtn = document.getElementById('sendMessage');
        const chatInput = document.getElementById('chatInput');

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        if (chatInput) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            chatInput.addEventListener('input', () => {
                this.updateSendButton();
            });
        }

        // Clear chat
        const clearBtn = document.getElementById('clearChat');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearChat());
        }

        // Model selection
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                window.app.showToast(`Modell gewechselt zu ${this.getModelName(e.target.value)}`, 'info');
            });
        }

        // Voice input
        const voiceBtn = document.getElementById('voiceInput');
        if (voiceBtn) {
            voiceBtn.addEventListener('click', () => this.toggleVoiceInput());
        }

        // File upload
        const fileBtn = document.getElementById('fileUpload');
        if (fileBtn) {
            fileBtn.addEventListener('click', () => this.openFileUpload());
        }
    }

    setupAutoResize() {
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                this.autoResizeTextarea(chatInput);
            });
        }
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    updateSendButton() {
        const sendBtn = document.getElementById('sendMessage');
        const chatInput = document.getElementById('chatInput');

        if (sendBtn && chatInput) {
            const hasContent = chatInput.value.trim().length > 0;
            sendBtn.disabled = !hasContent || this.isTyping;
            sendBtn.style.opacity = (hasContent && !this.isTyping) ? '1' : '0.5';
        }
    }

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        if (!chatInput) return;

        const message = chatInput.value.trim();
        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage('user', message);
        chatInput.value = '';
        this.autoResizeTextarea(chatInput);
        this.updateSendButton();

        // Save to history
        this.saveMessage('user', message);

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send to API
            const response = await this.sendToAPI(message);

            // Hide typing indicator
            this.hideTypingIndicator();

            // Add AI response
            this.addMessage('ai', response);
            this.saveMessage('ai', response);

            // Scroll to bottom
            this.scrollToBottom();

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('ai', 'Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.', 'error');
            console.error('Chat error:', error);
        }
    }

    async sendToAPI(message) {
        const endpoint = this.getAPIEndpoint();

        const requestData = {
            message: message,
            model: this.currentModel,
            timestamp: new Date().toISOString()
        };

        try {
            const response = await window.app.makeAPIRequest(endpoint, {
                method: 'POST',
                body: JSON.stringify(requestData)
            });

            return response.response || response.message || 'Keine Antwort erhalten';

        } catch (error) {
            // Fallback: Simulate AI response for demo
            if (endpoint.includes('demo')) {
                return this.generateDemoResponse(message);
            }
            throw error;
        }
    }

    generateDemoResponse(message) {
        const responses = [
            "Das ist eine interessante Frage. Lassen Sie mich darüber nachdenken...",
            "Ich verstehe Ihre Bedenken. Hier ist meine Einschätzung:",
            "Vielen Dank für Ihre Nachricht. Ich helfe Ihnen gerne weiter.",
            "Das ist ein komplexes Thema. Hier ist meine Analyse:",
            "Ich bin beeindruckt von Ihrer Frage. Lassen Sie mich antworten:"
        ];

        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        return `${randomResponse}\n\nIhre Nachricht: "${message}"\n\nAls Bundeskanzler KI würde ich sagen, dass dies ein wichtiges Thema ist, das sorgfältige Überlegung erfordert.`;
    }

    getAPIEndpoint() {
        // In production, this would be the actual API endpoint
        // For demo purposes, we'll use a mock endpoint
        return `/api/chat/${this.currentModel}`;
    }

    addMessage(sender, content, type = 'normal') {
        const messagesContainer = document.getElementById('chatMessages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const avatar = sender === 'user' ? 'user' : 'robot';
        const author = sender === 'user' ? 'Sie' : 'Bundeskanzler KI';

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-${avatar}"></i>
            </div>
            <div class="message-content ${type === 'error' ? 'error' : ''}">
                <div class="message-header">
                    <span class="message-author">${author}</span>
                    <span class="message-time">${this.formatTime(new Date())}</span>
                </div>
                <div class="message-text">${this.formatMessage(content)}</div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        this.messages.push({
            sender,
            content,
            timestamp: new Date(),
            type
        });
    }

    formatMessage(content) {
        // Basic formatting for line breaks and links
        return content
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>');
    }

    formatTime(date) {
        return date.toLocaleTimeString('de-DE', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.updateSendButton();

        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'flex';
        }
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.updateSendButton();

        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }, 100);
        }
    }

    clearChat() {
        if (!confirm('Möchten Sie wirklich den gesamten Chat-Verlauf löschen?')) {
            return;
        }

        const messagesContainer = document.getElementById('chatMessages');
        if (messagesContainer) {
            // Keep only the welcome message
            const welcomeMessage = messagesContainer.querySelector('.welcome-message');
            messagesContainer.innerHTML = '';
            if (welcomeMessage) {
                messagesContainer.appendChild(welcomeMessage);
            }
        }

        this.messages = [];
        localStorage.removeItem('chatHistory');

        window.app.showToast('Chat-Verlauf gelöscht', 'success');
    }

    saveMessage(sender, content) {
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        chatHistory.push({
            sender,
            content,
            timestamp: new Date().toISOString()
        });

        // Keep only last 100 messages
        if (chatHistory.length > 100) {
            chatHistory.splice(0, chatHistory.length - 100);
        }

        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }

    loadChatHistory() {
        try {
            const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');

            chatHistory.forEach(msg => {
                this.addMessage(msg.sender, msg.content);
            });

            if (chatHistory.length > 0) {
                this.scrollToBottom();
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    getModelName(modelKey) {
        const modelNames = {
            'bundeskanzler': 'Bundeskanzler KI',
            'multimodal': 'Multimodal KI',
            'fast': 'Schnell-Modus'
        };
        return modelNames[modelKey] || modelKey;
    }

    toggleVoiceInput() {
        // Voice input functionality would be implemented here
        window.app.showToast('Spracheingabe wird noch implementiert', 'info');
    }

    openFileUpload() {
        // File upload functionality would be implemented here
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*,.pdf,.txt,.doc,.docx';
        input.onchange = (e) => this.handleFileUpload(e.target.files[0]);
        input.click();
    }

    handleFileUpload(file) {
        if (!file) return;

        // File upload functionality would be implemented here
        window.app.showToast(`Datei "${file.name}" wird hochgeladen...`, 'info');

        // Simulate upload
        setTimeout(() => {
            window.app.showToast('Datei erfolgreich hochgeladen', 'success');
        }, 2000);
    }
}

// Initialize ChatManager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.ChatManager = new ChatManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatManager;
}