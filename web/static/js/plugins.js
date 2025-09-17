/**
 * Plugin-Manager für Bundeskanzler KI Web Interface
 * Verwaltet Plugin-Installation, -Konfiguration und -Status
 */

class PluginManager {
    constructor() {
        this.plugins = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Refresh plugins button
        const refreshBtn = document.getElementById('refreshPlugins');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadPlugins());
        }

        // Plugin modal
        const closeModalBtn = document.getElementById('closePluginModal');
        const cancelBtn = document.getElementById('cancelPluginConfig');

        if (closeModalBtn) {
            closeModalBtn.addEventListener('click', () => this.closePluginModal());
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.closePluginModal());
        }

        const saveBtn = document.getElementById('savePluginConfig');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.savePluginConfig());
        }

        // Click outside modal to close
        const modal = document.getElementById('pluginModal');
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closePluginModal();
                }
            });
        }
    }

    async loadPlugins() {
        try {
            window.app.showLoading(true);

            const response = await window.app.makeAPIRequest('/api/plugins/');
            this.plugins = response.data.loaded_plugins || {};
            this.availablePlugins = response.data.available_plugins || [];

            this.renderPlugins();
            this.updatePluginStats();

            window.app.showToast('Plugins erfolgreich geladen', 'success');

        } catch (error) {
            console.error('Error loading plugins:', error);
            this.showErrorState();
            window.app.showToast('Fehler beim Laden der Plugins', 'error');
        } finally {
            window.app.showLoading(false);
        }
    }

    renderPlugins() {
        const pluginsGrid = document.getElementById('pluginsGrid');
        if (!pluginsGrid) return;

        pluginsGrid.innerHTML = '';

        if (Object.keys(this.plugins).length === 0) {
            pluginsGrid.innerHTML = `
                <div class="no-plugins">
                    <i class="fas fa-puzzle-piece"></i>
                    <h3>Keine Plugins gefunden</h3>
                    <p>Es sind keine Plugins installiert oder verfügbar.</p>
                </div>
            `;
            return;
        }

        Object.entries(this.plugins).forEach(([name, plugin]) => {
            const pluginCard = this.createPluginCard(name, plugin);
            pluginsGrid.appendChild(pluginCard);
        });
    }

    createPluginCard(name, plugin) {
        const card = document.createElement('div');
        card.className = 'plugin-card';
        card.dataset.pluginName = name;

        const isActive = plugin.enabled;
        const statusClass = isActive ? 'active' : 'inactive';
        const statusText = isActive ? 'Aktiv' : 'Inaktiv';

        card.innerHTML = `
            <div class="plugin-header">
                <h3 class="plugin-title">${plugin.metadata.name}</h3>
                <span class="plugin-status ${statusClass}">${statusText}</span>
            </div>
            <p class="plugin-description">${plugin.metadata.description}</p>
            <div class="plugin-meta">
                <small>Version: ${plugin.metadata.version}</small>
                <small>Autor: ${plugin.metadata.author}</small>
            </div>
            <div class="plugin-actions">
                <button class="btn-secondary btn-small" onclick="window.PluginManager.togglePlugin('${name}')">
                    <i class="fas fa-${isActive ? 'stop' : 'play'}"></i>
                    ${isActive ? 'Stoppen' : 'Starten'}
                </button>
                <button class="btn-secondary btn-small" onclick="window.PluginManager.configurePlugin('${name}')">
                    <i class="fas fa-cog"></i>
                    Konfigurieren
                </button>
                <button class="btn-secondary btn-small" onclick="window.PluginManager.showPluginDetails('${name}')">
                    <i class="fas fa-info-circle"></i>
                    Details
                </button>
            </div>
        `;

        // Click handler for the entire card
        card.addEventListener('click', (e) => {
            if (!e.target.closest('.plugin-actions')) {
                this.showPluginDetails(name);
            }
        });

        return card;
    }

    updatePluginStats() {
        const activeCount = Object.values(this.plugins).filter(p => p.enabled).length;
        const totalCount = Object.keys(this.plugins).length;

        // Update any stats displays if they exist
        const statsElement = document.getElementById('pluginStats');
        if (statsElement) {
            statsElement.textContent = `${activeCount}/${totalCount} aktiv`;
        }
    }

    async togglePlugin(pluginName) {
        try {
            const plugin = this.plugins[pluginName];
            if (!plugin) return;

            const action = plugin.enabled ? 'disable' : 'enable';
            const endpoint = `/api/plugins/${pluginName}/${action}`;

            await window.app.makeAPIRequest(endpoint, { method: 'POST' });

            // Update local state
            plugin.enabled = !plugin.enabled;

            // Re-render plugins
            this.renderPlugins();

            const statusText = plugin.enabled ? 'aktiviert' : 'deaktiviert';
            window.app.showToast(`Plugin "${pluginName}" ${statusText}`, 'success');

        } catch (error) {
            console.error('Error toggling plugin:', error);
            window.app.showToast('Fehler beim Ändern des Plugin-Status', 'error');
        }
    }

    async configurePlugin(pluginName) {
        try {
            const plugin = this.plugins[pluginName];
            if (!plugin) return;

            // Get current configuration
            const configResponse = await window.app.makeAPIRequest(`/api/plugins/${pluginName}/config`);
            const currentConfig = configResponse;

            this.showPluginConfigModal(pluginName, plugin, currentConfig);

        } catch (error) {
            console.error('Error loading plugin config:', error);
            window.app.showToast('Fehler beim Laden der Plugin-Konfiguration', 'error');
        }
    }

    showPluginConfigModal(pluginName, plugin, config) {
        const modal = document.getElementById('pluginModal');
        const modalBody = document.getElementById('pluginModalBody');

        if (!modal || !modalBody) return;

        modalBody.innerHTML = `
            <div class="plugin-config-form">
                <div class="config-section">
                    <h4>Plugin-Informationen</h4>
                    <div class="config-item">
                        <label>Name:</label>
                        <span>${plugin.metadata.name}</span>
                    </div>
                    <div class="config-item">
                        <label>Version:</label>
                        <span>${plugin.metadata.version}</span>
                    </div>
                    <div class="config-item">
                        <label>Beschreibung:</label>
                        <span>${plugin.metadata.description}</span>
                    </div>
                </div>

                <div class="config-section">
                    <h4>Konfiguration</h4>
                    <div class="config-item">
                        <label for="pluginEnabled">
                            <input type="checkbox" id="pluginEnabled" ${config.enabled ? 'checked' : ''}>
                            Plugin aktivieren
                        </label>
                    </div>
                    <div class="config-item">
                        <label for="pluginPriority">Priorität:</label>
                        <input type="number" id="pluginPriority" value="${config.priority || 100}" min="0" max="1000">
                    </div>
                    <div class="config-item">
                        <label for="pluginAutoStart">
                            <input type="checkbox" id="pluginAutoStart" ${config.auto_start ? 'checked' : ''}>
                            Automatisch starten
                        </label>
                    </div>
                </div>

                <div class="config-section">
                    <h4>Erweiterte Einstellungen</h4>
                    <div id="advancedSettings">
                        <!-- Dynamische erweiterte Einstellungen werden hier geladen -->
                    </div>
                </div>
            </div>
        `;

        // Store current plugin name for saving
        modal.dataset.pluginName = pluginName;

        modal.style.display = 'flex';
    }

    async savePluginConfig() {
        const modal = document.getElementById('pluginModal');
        const pluginName = modal?.dataset.pluginName;

        if (!pluginName) return;

        try {
            const config = {
                enabled: document.getElementById('pluginEnabled')?.checked || false,
                priority: parseInt(document.getElementById('pluginPriority')?.value || '100'),
                auto_start: document.getElementById('pluginAutoStart')?.checked || false,
                settings: {} // Erweiterte Einstellungen würden hier hinzugefügt
            };

            await window.app.makeAPIRequest(`/api/plugins/${pluginName}/config`, {
                method: 'PUT',
                body: JSON.stringify(config)
            });

            // Update local state
            if (this.plugins[pluginName]) {
                this.plugins[pluginName].config = config;
                this.plugins[pluginName].enabled = config.enabled;
            }

            this.closePluginModal();
            this.renderPlugins();

            window.app.showToast('Plugin-Konfiguration gespeichert', 'success');

        } catch (error) {
            console.error('Error saving plugin config:', error);
            window.app.showToast('Fehler beim Speichern der Konfiguration', 'error');
        }
    }

    closePluginModal() {
        const modal = document.getElementById('pluginModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    showPluginDetails(pluginName) {
        const plugin = this.plugins[pluginName];
        if (!plugin) return;

        const detailsContainer = document.getElementById('pluginDetails');
        const titleElement = document.getElementById('pluginTitle');
        const versionElement = document.getElementById('pluginVersion');
        const authorElement = document.getElementById('pluginAuthor');
        const descriptionElement = document.getElementById('pluginDescription');
        const configElement = document.getElementById('pluginConfig');

        if (!detailsContainer || !titleElement) return;

        titleElement.textContent = plugin.metadata.name;
        versionElement.textContent = plugin.metadata.version;
        authorElement.textContent = plugin.metadata.author;
        descriptionElement.textContent = plugin.metadata.description;

        // Show configuration details
        configElement.innerHTML = `
            <div class="config-summary">
                <div class="config-item">
                    <strong>Status:</strong> ${plugin.enabled ? 'Aktiv' : 'Inaktiv'}
                </div>
                <div class="config-item">
                    <strong>Priorität:</strong> ${plugin.config?.priority || '100'}
                </div>
                <div class="config-item">
                    <strong>Autostart:</strong> ${plugin.config?.auto_start ? 'Ja' : 'Nein'}
                </div>
                <div class="config-item">
                    <strong>Typ:</strong> ${plugin.type}
                </div>
            </div>
        `;

        detailsContainer.style.display = 'block';
        detailsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    showErrorState() {
        const pluginsGrid = document.getElementById('pluginsGrid');
        if (!pluginsGrid) return;

        pluginsGrid.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Fehler beim Laden der Plugins</h3>
                <p>Bitte versuchen Sie es später erneut.</p>
                <button class="btn-primary" onclick="window.PluginManager.loadPlugins()">
                    <i class="fas fa-sync"></i> Erneut versuchen
                </button>
            </div>
        `;
    }

    async installPlugin(pluginName) {
        try {
            await window.app.makeAPIRequest(`/api/plugins/${pluginName}/load`, {
                method: 'POST'
            });

            window.app.showToast(`Plugin "${pluginName}" installiert`, 'success');
            this.loadPlugins(); // Reload plugin list

        } catch (error) {
            console.error('Error installing plugin:', error);
            window.app.showToast('Fehler bei der Plugin-Installation', 'error');
        }
    }

    async uninstallPlugin(pluginName) {
        if (!confirm(`Möchten Sie das Plugin "${pluginName}" wirklich deinstallieren?`)) {
            return;
        }

        try {
            await window.app.makeAPIRequest(`/api/plugins/${pluginName}/unload`, {
                method: 'POST'
            });

            window.app.showToast(`Plugin "${pluginName}" deinstalliert`, 'success');
            this.loadPlugins(); // Reload plugin list

        } catch (error) {
            console.error('Error uninstalling plugin:', error);
            window.app.showToast('Fehler bei der Plugin-Deinstallation', 'error');
        }
    }
}

// Initialize PluginManager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.PluginManager = new PluginManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PluginManager;
}