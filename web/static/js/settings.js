/**
 * Einstellungs-Manager für Bundeskanzler KI Web Interface
 * Verwaltet System-Konfiguration, Benutzereinstellungen und API-Schlüssel
 */

class SettingsManager {
    constructor() {
        this.settings = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadSettings();
    }

    setupEventListeners() {
        // Save settings button
        const saveBtn = document.getElementById('saveSettings');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveSettings());
        }

        // Reset settings button
        const resetBtn = document.getElementById('resetSettings');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetSettings());
        }

        // Export settings button
        const exportBtn = document.getElementById('exportSettings');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportSettings());
        }

        // Import settings button
        const importBtn = document.getElementById('importSettings');
        if (importBtn) {
            importBtn.addEventListener('click', () => this.triggerImport());
        }

        // File input for import
        const importFile = document.getElementById('importSettingsFile');
        if (importFile) {
            importFile.addEventListener('change', (e) => this.importSettings(e.target.files[0]));
        }

        // Test API connection button
        const testApiBtn = document.getElementById('testApiConnection');
        if (testApiBtn) {
            testApiBtn.addEventListener('click', () => this.testApiConnection());
        }

        // Generate API key button
        const generateApiKeyBtn = document.getElementById('generateApiKey');
        if (generateApiKeyBtn) {
            generateApiKeyBtn.addEventListener('click', () => this.generateApiKey());
        }

        // Theme change handler
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.addEventListener('change', (e) => {
                window.app.setTheme(e.target.value);
            });
        }

        // Language change handler
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.addEventListener('change', (e) => {
                this.changeLanguage(e.target.value);
            });
        }
    }

    async loadSettings() {
        try {
            window.app.showLoading(true);

            const response = await window.app.makeAPIRequest('/api/settings/');
            this.settings = response;

            this.populateSettingsForm();
            this.updateSettingsDisplay();

            window.app.showToast('Einstellungen erfolgreich geladen', 'success');

        } catch (error) {
            console.error('Error loading settings:', error);
            this.showErrorState();
            window.app.showToast('Fehler beim Laden der Einstellungen', 'error');
        } finally {
            window.app.showLoading(false);
        }
    }

    populateSettingsForm() {
        // General Settings
        this.setFormValue('systemName', this.settings.system_name || 'Bundeskanzler KI');
        this.setFormValue('maxTokens', this.settings.max_tokens || 2048);
        this.setFormValue('temperature', this.settings.temperature || 0.7);
        this.setFormValue('topP', this.settings.top_p || 0.9);
        this.setFormValue('frequencyPenalty', this.settings.frequency_penalty || 0.0);
        this.setFormValue('presencePenalty', this.settings.presence_penalty || 0.0);

        // API Settings
        this.setFormValue('apiHost', this.settings.api_host || 'localhost');
        this.setFormValue('apiPort', this.settings.api_port || 8000);
        this.setFormValue('apiKey', this.settings.api_key || '');
        this.setFormValue('useHttps', this.settings.use_https || false);

        // Performance Settings
        this.setFormValue('enableCaching', this.settings.enable_caching || true);
        this.setFormValue('cacheSize', this.settings.cache_size || 1000);
        this.setFormValue('enableBatching', this.settings.enable_batching || true);
        this.setFormValue('batchSize', this.settings.batch_size || 10);
        this.setFormValue('enableAutoScaling', this.settings.enable_auto_scaling || true);

        // Security Settings
        this.setFormValue('enableRateLimiting', this.settings.enable_rate_limiting || true);
        this.setFormValue('rateLimitRequests', this.settings.rate_limit_requests || 100);
        this.setFormValue('rateLimitWindow', this.settings.rate_limit_window || 60);
        this.setFormValue('enableLogging', this.settings.enable_logging || true);
        this.setFormValue('logLevel', this.settings.log_level || 'INFO');

        // UI Settings
        this.setFormValue('themeSelect', this.settings.theme || 'light');
        this.setFormValue('languageSelect', this.settings.language || 'de');
        this.setFormValue('enableNotifications', this.settings.enable_notifications || true);
        this.setFormValue('autoRefreshInterval', this.settings.auto_refresh_interval || 30);

        // Plugin Settings
        this.setFormValue('enablePlugins', this.settings.enable_plugins || true);
        this.setFormValue('pluginAutoLoad', this.settings.plugin_auto_load || false);
        this.setFormValue('maxPluginInstances', this.settings.max_plugin_instances || 10);
    }

    setFormValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (!element) return;

        if (element.type === 'checkbox') {
            element.checked = value;
        } else {
            element.value = value;
        }
    }

    getFormValue(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return null;

        if (element.type === 'checkbox') {
            return element.checked;
        } else if (element.type === 'number') {
            return parseFloat(element.value) || 0;
        } else {
            return element.value;
        }
    }

    updateSettingsDisplay() {
        // Update any display elements that show current settings
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = this.settings.theme || 'light';
        }

        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.value = this.settings.language || 'de';
        }
    }

    async saveSettings() {
        try {
            window.app.showLoading(true);

            const newSettings = {
                // General
                system_name: this.getFormValue('systemName'),
                max_tokens: this.getFormValue('maxTokens'),
                temperature: this.getFormValue('temperature'),
                top_p: this.getFormValue('topP'),
                frequency_penalty: this.getFormValue('frequencyPenalty'),
                presence_penalty: this.getFormValue('presencePenalty'),

                // API
                api_host: this.getFormValue('apiHost'),
                api_port: this.getFormValue('apiPort'),
                api_key: this.getFormValue('apiKey'),
                use_https: this.getFormValue('useHttps'),

                // Performance
                enable_caching: this.getFormValue('enableCaching'),
                cache_size: this.getFormValue('cacheSize'),
                enable_batching: this.getFormValue('enableBatching'),
                batch_size: this.getFormValue('batchSize'),
                enable_auto_scaling: this.getFormValue('enableAutoScaling'),

                // Security
                enable_rate_limiting: this.getFormValue('enableRateLimiting'),
                rate_limit_requests: this.getFormValue('rateLimitRequests'),
                rate_limit_window: this.getFormValue('rateLimitWindow'),
                enable_logging: this.getFormValue('enableLogging'),
                log_level: this.getFormValue('logLevel'),

                // UI
                theme: this.getFormValue('themeSelect'),
                language: this.getFormValue('languageSelect'),
                enable_notifications: this.getFormValue('enableNotifications'),
                auto_refresh_interval: this.getFormValue('autoRefreshInterval'),

                // Plugins
                enable_plugins: this.getFormValue('enablePlugins'),
                plugin_auto_load: this.getFormValue('pluginAutoLoad'),
                max_plugin_instances: this.getFormValue('maxPluginInstances')
            };

            await window.app.makeAPIRequest('/api/settings/', {
                method: 'PUT',
                body: JSON.stringify(newSettings)
            });

            this.settings = { ...this.settings, ...newSettings };
            window.app.showToast('Einstellungen erfolgreich gespeichert', 'success');

        } catch (error) {
            console.error('Error saving settings:', error);
            window.app.showToast('Fehler beim Speichern der Einstellungen', 'error');
        } finally {
            window.app.showLoading(false);
        }
    }

    async resetSettings() {
        if (!confirm('Möchten Sie wirklich alle Einstellungen auf Standardwerte zurücksetzen?')) {
            return;
        }

        try {
            window.app.showLoading(true);

            await window.app.makeAPIRequest('/api/settings/reset', {
                method: 'POST'
            });

            await this.loadSettings();
            window.app.showToast('Einstellungen erfolgreich zurückgesetzt', 'success');

        } catch (error) {
            console.error('Error resetting settings:', error);
            window.app.showToast('Fehler beim Zurücksetzen der Einstellungen', 'error');
        } finally {
            window.app.showLoading(false);
        }
    }

    async exportSettings() {
        try {
            const settingsJson = JSON.stringify(this.settings, null, 2);
            this.downloadFile(settingsJson, 'bundeskanzler_settings.json', 'application/json');

            window.app.showToast('Einstellungen erfolgreich exportiert', 'success');

        } catch (error) {
            console.error('Error exporting settings:', error);
            window.app.showToast('Fehler beim Exportieren der Einstellungen', 'error');
        }
    }

    triggerImport() {
        const importFile = document.getElementById('importSettingsFile');
        if (importFile) {
            importFile.click();
        }
    }

    async importSettings(file) {
        if (!file) return;

        try {
            const text = await file.text();
            const importedSettings = JSON.parse(text);

            // Validate imported settings
            if (!this.validateSettings(importedSettings)) {
                throw new Error('Ungültige Einstellungsdatei');
            }

            await window.app.makeAPIRequest('/api/settings/import', {
                method: 'POST',
                body: JSON.stringify(importedSettings)
            });

            this.settings = importedSettings;
            this.populateSettingsForm();
            this.updateSettingsDisplay();

            window.app.showToast('Einstellungen erfolgreich importiert', 'success');

        } catch (error) {
            console.error('Error importing settings:', error);
            window.app.showToast('Fehler beim Importieren der Einstellungen', 'error');
        }
    }

    validateSettings(settings) {
        // Basic validation - check for required fields
        const requiredFields = ['system_name', 'max_tokens', 'temperature'];
        return requiredFields.every(field => settings.hasOwnProperty(field));
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);
    }

    async testApiConnection() {
        const testBtn = document.getElementById('testApiConnection');
        const originalText = testBtn.textContent;

        try {
            testBtn.textContent = 'Teste...';
            testBtn.disabled = true;

            const response = await window.app.makeAPIRequest('/api/health');

            if (response.status === 'healthy') {
                window.app.showToast('API-Verbindung erfolgreich', 'success');
            } else {
                throw new Error('API nicht verfügbar');
            }

        } catch (error) {
            console.error('API connection test failed:', error);
            window.app.showToast('API-Verbindung fehlgeschlagen', 'error');
        } finally {
            testBtn.textContent = originalText;
            testBtn.disabled = false;
        }
    }

    async generateApiKey() {
        try {
            const response = await window.app.makeAPIRequest('/api/settings/generate-api-key', {
                method: 'POST'
            });

            const newApiKey = response.api_key;
            this.setFormValue('apiKey', newApiKey);
            this.settings.api_key = newApiKey;

            window.app.showToast('Neuer API-Schlüssel generiert', 'success');

        } catch (error) {
            console.error('Error generating API key:', error);
            window.app.showToast('Fehler beim Generieren des API-Schlüssels', 'error');
        }
    }

    changeLanguage(language) {
        // This would typically reload the UI with the new language
        // For now, just save the setting
        this.settings.language = language;
        window.app.showToast(`Sprache auf ${language === 'de' ? 'Deutsch' : 'English'} geändert`, 'info');
    }

    showErrorState() {
        const settingsForm = document.getElementById('settingsForm');
        if (settingsForm) {
            settingsForm.innerHTML = `
                <div class="error-state">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h3>Fehler beim Laden der Einstellungen</h3>
                    <p>Bitte versuchen Sie es später erneut.</p>
                    <button class="btn-primary" onclick="window.SettingsManager.loadSettings()">
                        <i class="fas fa-sync"></i> Erneut versuchen
                    </button>
                </div>
            `;
        }
    }
}

// Initialize SettingsManager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.SettingsManager = new SettingsManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SettingsManager;
}