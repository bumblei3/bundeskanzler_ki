/**
 * Monitoring-System für Bundeskanzler KI Web Interface
 * Zeigt System-Performance, Ressourcen-Nutzung und Logs in Echtzeit
 */

class MonitoringManager {
    constructor() {
        this.charts = {};
        this.updateInterval = null;
        this.isAutoRefresh = true;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refreshMonitoring');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefreshToggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                this.isAutoRefresh = e.target.checked;
                if (this.isAutoRefresh) {
                    this.startAutoRefresh();
                } else {
                    this.stopAutoRefresh();
                }
            });
        }

        // Time range selector
        const timeRangeSelect = document.getElementById('timeRangeSelect');
        if (timeRangeSelect) {
            timeRangeSelect.addEventListener('change', (e) => {
                this.changeTimeRange(e.target.value);
            });
        }

        // Export logs button
        const exportLogsBtn = document.getElementById('exportLogs');
        if (exportLogsBtn) {
            exportLogsBtn.addEventListener('click', () => this.exportLogs());
        }

        // Clear logs button
        const clearLogsBtn = document.getElementById('clearLogs');
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener('click', () => this.clearLogs());
        }
    }

    initializeCharts() {
        // CPU Usage Chart
        this.charts.cpu = this.createChart('cpuChart', 'CPU-Auslastung', '%', '#3498db');

        // Memory Usage Chart
        this.charts.memory = this.createChart('memoryChart', 'Speicher-Auslastung', '%', '#e74c3c');

        // GPU Usage Chart (if available)
        this.charts.gpu = this.createChart('gpuChart', 'GPU-Auslastung', '%', '#9b59b6');

        // Request Rate Chart
        this.charts.requests = this.createChart('requestsChart', 'Anfragen pro Minute', 'req/min', '#2ecc71');

        // Response Time Chart
        this.charts.responseTime = this.createChart('responseTimeChart', 'Antwortzeit', 'ms', '#f39c12');
    }

    createChart(containerId, label, unit, color) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        const canvas = document.createElement('canvas');
        container.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: label,
                    data: [],
                    borderColor: color,
                    backgroundColor: color + '20',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            displayFormats: {
                                minute: 'HH:mm',
                                hour: 'HH:mm'
                            }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + unit;
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y + unit;
                            }
                        }
                    }
                },
                animation: {
                    duration: 500
                }
            }
        });

        return chart;
    }

    startAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.updateInterval = setInterval(() => {
            if (this.isAutoRefresh) {
                this.refreshData();
            }
        }, 5000); // Update every 5 seconds
    }

    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    async refreshData() {
        try {
            window.app.showLoading(true);

            // Load system metrics
            const metricsResponse = await window.app.makeAPIRequest('/api/monitoring/metrics');
            const metrics = metricsResponse;

            // Load logs
            const logsResponse = await window.app.makeAPIRequest('/api/monitoring/logs?limit=100');
            const logs = logsResponse.logs || [];

            this.updateMetrics(metrics);
            this.updateLogs(logs);
            this.updateSystemInfo(metrics);

        } catch (error) {
            console.error('Error refreshing monitoring data:', error);
            window.app.showToast('Fehler beim Aktualisieren der Monitoring-Daten', 'error');
        } finally {
            window.app.showLoading(false);
        }
    }

    updateMetrics(metrics) {
        const now = new Date();

        // Update CPU chart
        if (this.charts.cpu && metrics.cpu_usage !== undefined) {
            this.updateChart(this.charts.cpu, now, metrics.cpu_usage);
        }

        // Update Memory chart
        if (this.charts.memory && metrics.memory_usage !== undefined) {
            this.updateChart(this.charts.memory, now, metrics.memory_usage);
        }

        // Update GPU chart
        if (this.charts.gpu && metrics.gpu_usage !== undefined) {
            this.updateChart(this.charts.gpu, now, metrics.gpu_usage);
        }

        // Update Requests chart
        if (this.charts.requests && metrics.request_rate !== undefined) {
            this.updateChart(this.charts.requests, now, metrics.request_rate);
        }

        // Update Response Time chart
        if (this.charts.responseTime && metrics.avg_response_time !== undefined) {
            this.updateChart(this.charts.responseTime, now, metrics.avg_response_time);
        }

        // Update metric displays
        this.updateMetricDisplays(metrics);
    }

    updateChart(chart, time, value) {
        if (!chart) return;

        chart.data.labels.push(time);
        chart.data.datasets[0].data.push(value);

        // Keep only last 50 data points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update('none'); // Update without animation for performance
    }

    updateMetricDisplays(metrics) {
        // CPU Usage
        const cpuElement = document.getElementById('cpuUsage');
        if (cpuElement && metrics.cpu_usage !== undefined) {
            cpuElement.textContent = `${metrics.cpu_usage.toFixed(1)}%`;
            this.updateProgressBar('cpuProgress', metrics.cpu_usage);
        }

        // Memory Usage
        const memoryElement = document.getElementById('memoryUsage');
        if (memoryElement && metrics.memory_usage !== undefined) {
            memoryElement.textContent = `${metrics.memory_usage.toFixed(1)}%`;
            this.updateProgressBar('memoryProgress', metrics.memory_usage);
        }

        // GPU Usage
        const gpuElement = document.getElementById('gpuUsage');
        if (gpuElement && metrics.gpu_usage !== undefined) {
            gpuElement.textContent = `${metrics.gpu_usage.toFixed(1)}%`;
            this.updateProgressBar('gpuProgress', metrics.gpu_usage);
        }

        // Active Connections
        const connectionsElement = document.getElementById('activeConnections');
        if (connectionsElement && metrics.active_connections !== undefined) {
            connectionsElement.textContent = metrics.active_connections;
        }

        // Total Requests
        const totalRequestsElement = document.getElementById('totalRequests');
        if (totalRequestsElement && metrics.total_requests !== undefined) {
            totalRequestsElement.textContent = metrics.total_requests.toLocaleString();
        }

        // Average Response Time
        const avgResponseTimeElement = document.getElementById('avgResponseTime');
        if (avgResponseTimeElement && metrics.avg_response_time !== undefined) {
            avgResponseTimeElement.textContent = `${metrics.avg_response_time.toFixed(0)}ms`;
        }
    }

    updateProgressBar(barId, percentage) {
        const bar = document.getElementById(barId);
        if (bar) {
            bar.style.width = `${Math.min(percentage, 100)}%`;

            // Update color based on usage level
            if (percentage >= 90) {
                bar.style.backgroundColor = '#e74c3c'; // Red
            } else if (percentage >= 70) {
                bar.style.backgroundColor = '#f39c12'; // Orange
            } else {
                bar.style.backgroundColor = '#2ecc71'; // Green
            }
        }
    }

    updateSystemInfo(metrics) {
        // Update uptime
        const uptimeElement = document.getElementById('systemUptime');
        if (uptimeElement && metrics.uptime_seconds !== undefined) {
            uptimeElement.textContent = this.formatUptime(metrics.uptime_seconds);
        }

        // Update Python version
        const pythonVersionElement = document.getElementById('pythonVersion');
        if (pythonVersionElement && metrics.python_version) {
            pythonVersionElement.textContent = metrics.python_version;
        }

        // Update model info
        const modelInfoElement = document.getElementById('modelInfo');
        if (modelInfoElement && metrics.model_info) {
            modelInfoElement.textContent = metrics.model_info;
        }
    }

    formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);

        if (days > 0) {
            return `${days}d ${hours}h ${minutes}m`;
        } else if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }

    updateLogs(logs) {
        const logsContainer = document.getElementById('logsContainer');
        if (!logsContainer) return;

        logsContainer.innerHTML = '';

        if (logs.length === 0) {
            logsContainer.innerHTML = '<div class="no-logs">Keine Logs verfügbar</div>';
            return;
        }

        logs.forEach(log => {
            const logEntry = this.createLogEntry(log);
            logsContainer.appendChild(logEntry);
        });

        // Auto-scroll to bottom
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    createLogEntry(log) {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${log.level.toLowerCase()}`;

        const timestamp = new Date(log.timestamp).toLocaleString('de-DE');
        const level = log.level.toUpperCase();

        entry.innerHTML = `
            <span class="log-timestamp">${timestamp}</span>
            <span class="log-level">${level}</span>
            <span class="log-message">${this.escapeHtml(log.message)}</span>
        `;

        return entry;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    changeTimeRange(range) {
        // This would typically reload data with different time ranges
        // For now, just refresh current data
        this.refreshData();
    }

    async exportLogs() {
        try {
            const logsResponse = await window.app.makeAPIRequest('/api/monitoring/logs?limit=1000');
            const logs = logsResponse.logs || [];

            const csvContent = this.convertLogsToCSV(logs);
            this.downloadFile(csvContent, 'bundeskanzler_logs.csv', 'text/csv');

            window.app.showToast('Logs erfolgreich exportiert', 'success');

        } catch (error) {
            console.error('Error exporting logs:', error);
            window.app.showToast('Fehler beim Exportieren der Logs', 'error');
        }
    }

    convertLogsToCSV(logs) {
        const headers = ['Timestamp', 'Level', 'Message'];
        const rows = logs.map(log => [
            log.timestamp,
            log.level,
            `"${log.message.replace(/"/g, '""')}"` // Escape quotes
        ]);

        return [headers, ...rows].map(row => row.join(',')).join('\n');
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

    async clearLogs() {
        if (!confirm('Möchten Sie wirklich alle Logs löschen?')) {
            return;
        }

        try {
            await window.app.makeAPIRequest('/api/monitoring/logs', {
                method: 'DELETE'
            });

            this.updateLogs([]);
            window.app.showToast('Logs erfolgreich gelöscht', 'success');

        } catch (error) {
            console.error('Error clearing logs:', error);
            window.app.showToast('Fehler beim Löschen der Logs', 'error');
        }
    }

    destroy() {
        this.stopAutoRefresh();
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
    }
}

// Initialize MonitoringManager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.MonitoringManager = new MonitoringManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MonitoringManager;
}