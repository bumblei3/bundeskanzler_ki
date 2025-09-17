/**
 * Bundeskanzler KI Web Interface - Haupt-JavaScript
 * Verwaltet Navigation, Theme und grundlegende Funktionalität
 */

class BundeskanzlerApp {
    constructor() {
        this.currentTab = 'chat';
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
        this.applyTheme();
        this.showTab(this.currentTab);
        this.updateConnectionStatus();
        this.setupPeriodicUpdates();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.dataset.tab;
                this.showTab(tab);
            });
        });

        // Theme toggle (wird später implementiert)
        document.getElementById('themeSelect')?.addEventListener('change', (e) => {
            this.setTheme(e.target.value);
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Online/Offline status
        window.addEventListener('online', () => this.updateConnectionStatus());
        window.addEventListener('offline', () => this.updateConnectionStatus());
    }

    showTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        this.currentTab = tabName;

        // Trigger tab-specific initialization
        this.onTabChange(tabName);
    }

    onTabChange(tabName) {
        switch(tabName) {
            case 'chat':
                if (window.ChatManager) window.ChatManager.init();
                break;
            case 'plugins':
                if (window.PluginManager) window.PluginManager.loadPlugins();
                break;
            case 'monitoring':
                if (window.MonitoringManager) window.MonitoringManager.loadMetrics();
                break;
            case 'settings':
                if (window.SettingsManager) window.SettingsManager.loadSettings();
                break;
        }
    }

    setTheme(theme) {
        this.theme = theme;
        localStorage.setItem('theme', theme);
        this.applyTheme();

        this.showToast('Theme aktualisiert', 'success');
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);

        // Update theme selector
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = this.theme;
        }
    }

    updateConnectionStatus() {
        const statusBtn = document.getElementById('connectionStatus');
        const isOnline = navigator.onLine;

        if (statusBtn) {
            statusBtn.innerHTML = `
                <i class="fas fa-${isOnline ? 'circle' : 'exclamation-triangle'}"></i>
                ${isOnline ? 'Verbunden' : 'Offline'}
            `;
            statusBtn.style.color = isOnline ? 'var(--success-color)' : 'var(--error-color)';
        }
    }

    setupPeriodicUpdates() {
        // Update connection status every 30 seconds
        setInterval(() => {
            this.updateConnectionStatus();
        }, 30000);

        // Update monitoring data every 60 seconds
        setInterval(() => {
            if (this.currentTab === 'monitoring' && window.MonitoringManager) {
                window.MonitoringManager.loadMetrics();
            }
        }, 60000);
    }

    handleResize() {
        // Handle responsive layout changes
        const isMobile = window.innerWidth <= 768;

        if (isMobile) {
            // Mobile-specific adjustments
            this.adjustForMobile();
        }
    }

    adjustForMobile() {
        // Mobile layout adjustments
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            const height = window.innerHeight - 120; // Header + padding
            chatContainer.style.height = `${height}px`;
        }
    }

    loadSettings() {
        // Load user settings from localStorage
        const settings = {
            theme: localStorage.getItem('theme') || 'light',
            language: localStorage.getItem('language') || 'de',
            defaultModel: localStorage.getItem('defaultModel') || 'bundeskanzler'
        };

        // Apply settings
        this.setTheme(settings.theme);

        // Update form elements
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) languageSelect.value = settings.language;

        const defaultModelSelect = document.getElementById('defaultModel');
        if (defaultModelSelect) defaultModelSelect.value = settings.defaultModel;
    }

    saveSettings() {
        const settings = {
            theme: document.getElementById('themeSelect')?.value || 'light',
            language: document.getElementById('languageSelect')?.value || 'de',
            defaultModel: document.getElementById('defaultModel')?.value || 'bundeskanzler'
        };

        Object.entries(settings).forEach(([key, value]) => {
            localStorage.setItem(key, value);
        });

        this.showToast('Einstellungen gespeichert', 'success');
    }

    showToast(message, type = 'info', title = '') {
        const toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const iconMap = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };

        toast.innerHTML = `
            <i class="fas fa-${iconMap[type] || 'info-circle'}"></i>
            <div class="toast-content">
                <div class="toast-title">${title || type.charAt(0).toUpperCase() + type.slice(1)}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        toastContainer.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);

        // Close button
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });
    }

    showLoading(show = true) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    async makeAPIRequest(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const finalOptions = { ...defaultOptions, ...options };

        try {
            this.showLoading(true);
            const response = await fetch(endpoint, finalOptions);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            this.showToast(`API-Fehler: ${error.message}`, 'error');
            throw error;
        } finally {
            this.showLoading(false);
        }
    }

    formatDate(date) {
        return new Intl.DateTimeFormat('de-DE', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    }

    formatNumber(num) {
        return new Intl.NumberFormat('de-DE').format(num);
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
}

// Global error handler
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    if (window.app) {
        window.app.showToast('Ein unerwarteter Fehler ist aufgetreten', 'error');
    }
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    if (window.app) {
        window.app.showToast('Ein Netzwerkfehler ist aufgetreten', 'error');
    }
});

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new BundeskanzlerApp();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BundeskanzlerApp;
}