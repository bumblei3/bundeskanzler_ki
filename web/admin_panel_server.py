#!/usr/bin/env python3
"""
Einfaches Admin-Panel für die Bundeskanzler KI
Startet einen lokalen HTTP-Server mit Admin-Interface
"""

import http.server
import json
import socketserver
import urllib.parse
import webbrowser
from pathlib import Path

import requests


class AdminPanelHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/admin":
            self.send_admin_panel()
        elif self.path.startswith("/api/"):
            self.handle_api_request()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.handle_api_request()
        else:
            self.send_response(404)
            self.end_headers()

    def send_admin_panel(self):
        html = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔐 Bundeskanzler KI - Admin Panel</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f6fa; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .btn { padding: 10px 20px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #5a67d8; }
        .btn-danger { background: #e53e3e; }
        .btn-danger:hover { background: #c53030; }
        .status { padding: 5px 10px; border-radius: 15px; font-size: 12px; font-weight: bold; }
        .status-ok { background: #c6f6d5; color: #22543d; }
        .status-error { background: #fed7d7; color: #742a2a; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; padding: 10px 15px; background: #edf2f7; border-radius: 5px; }
        .log-entry { font-family: 'Courier New', monospace; font-size: 12px; margin: 5px 0; padding: 8px; background: #f7fafc; border-left: 3px solid #cbd5e0; }
        .user-row { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #e2e8f0; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; }
        #loginForm { max-width: 400px; margin: 50px auto; }
        #adminPanel { display: none; }
        .tabs { display: flex; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; }
        .tab.active { border-bottom: 2px solid #667eea; color: #667eea; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <!-- Login Form -->
    <div id="loginForm" class="card">
        <h2 style="text-align: center; margin-bottom: 20px;">🔐 Admin Login</h2>
        <div class="form-group">
            <label>Username:</label>
            <input type="text" id="username" value="admin">
        </div>
        <div class="form-group">
            <label>Password:</label>
            <input type="password" id="password" value="admin123!">
        </div>
        <button class="btn" onclick="login()" style="width: 100%;">Login</button>
        <div id="loginStatus" style="margin-top: 10px; text-align: center;"></div>
    </div>

    <!-- Admin Panel -->
    <div id="adminPanel">
        <div class="header">
            <h1>🔐 Bundeskanzler KI - Admin Panel</h1>
            <p>Vollständige Systemverwaltung</p>
        </div>

        <div class="container">
            <div class="tabs">
                <button class="tab active" onclick="showTab('dashboard')">📊 Dashboard</button>
                <button class="tab" onclick="showTab('users')">👥 Benutzer</button>
                <button class="tab" onclick="showTab('logs')">📋 Logs</button>
                <button class="tab" onclick="showTab('memory')">💾 Memory</button>
            </div>

            <!-- Dashboard Tab -->
            <div id="dashboard" class="tab-content active">
                <div class="card">
                    <h3>System Status</h3>
                    <button class="btn" onclick="loadDashboard()">🔄 Refresh</button>
                    <div id="dashboardData"></div>
                </div>
            </div>

            <!-- Users Tab -->
            <div id="users" class="tab-content">
                <div class="card">
                    <h3>Benutzer-Management</h3>
                    <button class="btn" onclick="loadUsers()">👥 Benutzer laden</button>
                    <div id="usersData"></div>
                </div>
            </div>

            <!-- Logs Tab -->
            <div id="logs" class="tab-content">
                <div class="card">
                    <h3>Log Viewer</h3>
                    <div class="form-group">
                        <label>Log-Datei:</label>
                        <select id="logType">
                            <option value="api.log">API Logs</option>
                            <option value="memory.log">Memory Logs</option>
                            <option value="errors.log">Error Logs</option>
                        </select>
                    </div>
                    <button class="btn" onclick="loadLogs()">📋 Logs laden</button>
                    <div id="logsData"></div>
                </div>
            </div>

            <!-- Memory Tab -->
            <div id="memory" class="tab-content">
                <div class="card">
                    <h3>Memory Management</h3>
                    <button class="btn" onclick="loadMemoryStats()">📊 Memory Stats</button>
                    <div id="memoryData"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let adminToken = null;
        const API_URL = 'http://localhost:8000';

        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const formData = new FormData();
                formData.append('username', username);
                formData.append('password', password);
                
                const response = await fetch(`${API_URL}/auth/admin-token`, {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const data = await response.json();
                    adminToken = data.access_token;
                    document.getElementById('loginForm').style.display = 'none';
                    document.getElementById('adminPanel').style.display = 'block';
                    document.getElementById('loginStatus').innerHTML = '<span class="status status-ok">✅ Login erfolgreich!</span>';
                    loadDashboard(); // Auto-load dashboard
                } else {
                    document.getElementById('loginStatus').innerHTML = '<span class="status status-error">❌ Login fehlgeschlagen!</span>';
                }
            } catch (error) {
                document.getElementById('loginStatus').innerHTML = '<span class="status status-error">❌ Verbindungsfehler!</span>';
                console.error('Login error:', error);
            }
        }

        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        async function loadDashboard() {
            try {
                const response = await fetch(`${API_URL}/admin/health`, {
                    headers: { 'Authorization': `Bearer ${adminToken}` }
                });
                
                if (response.ok) {
                    const health = await response.json();
                    const html = `
                        <div class="metric">
                            <strong>Uptime:</strong> ${health.system.uptime.toFixed(1)}s
                        </div>
                        <div class="metric">
                            <strong>Requests:</strong> ${health.system.request_count}
                        </div>
                        <div class="metric">
                            <strong>Components:</strong> ${health.system.components_initialized ? '✅ OK' : '❌ Error'}
                        </div>
                        <div class="metric">
                            <strong>Files:</strong> ${health.files.logs_accessible ? '✅ OK' : '❌ Error'}
                        </div>
                    `;
                    document.getElementById('dashboardData').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('dashboardData').innerHTML = '<span class="status status-error">Fehler beim Laden</span>';
            }
        }

        async function loadUsers() {
            try {
                const response = await fetch(`${API_URL}/admin/users`, {
                    headers: { 'Authorization': `Bearer ${adminToken}` }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    let html = `<h4>${data.total} Benutzer gefunden:</h4>`;
                    
                    data.users.forEach(user => {
                        const adminBadge = user.is_admin ? '🔐 Admin' : '👤 User';
                        const statusBadge = user.is_active ? '✅' : '❌';
                        html += `
                            <div class="user-row">
                                <div>
                                    <strong>${user.user_id}</strong> (${user.email})<br>
                                    <small>${adminBadge} | Login Count: ${user.login_count}</small>
                                </div>
                                <div class="status ${user.is_active ? 'status-ok' : 'status-error'}">
                                    ${statusBadge} ${user.is_active ? 'Aktiv' : 'Inaktiv'}
                                </div>
                            </div>
                        `;
                    });
                    
                    document.getElementById('usersData').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('usersData').innerHTML = '<span class="status status-error">Fehler beim Laden</span>';
            }
        }

        async function loadLogs() {
            const logType = document.getElementById('logType').value;
            
            try {
                const response = await fetch(`${API_URL}/admin/logs/${logType}?lines=10`, {
                    headers: { 'Authorization': `Bearer ${adminToken}` }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    let html = `<h4>${logType} (${data.entries.length} Einträge):</h4>`;
                    
                    data.entries.reverse().forEach(entry => {
                        html += `
                            <div class="log-entry">
                                <strong>${entry.timestamp}</strong> [${entry.level}] ${entry.logger}<br>
                                ${entry.message}
                            </div>
                        `;
                    });
                    
                    document.getElementById('logsData').innerHTML = html;
                }
            } catch (error) {
                document.getElementById('logsData').innerHTML = '<span class="status status-error">Fehler beim Laden</span>';
            }
        }

        async function loadMemoryStats() {
            try {
                const response = await fetch(`${API_URL}/admin/memory/stats`, {
                    headers: { 'Authorization': `Bearer ${adminToken}` }
                });
                
                if (response.ok) {
                    const stats = await response.json();
                    const html = `
                        <div class="metric">
                            <strong>Kurzzeitgedächtnis:</strong> ${stats.kurzzeitgedaechtnis_entries || 0}
                        </div>
                        <div class="metric">
                            <strong>Langzeitgedächtnis:</strong> ${stats.langzeitgedaechtnis_entries || 0}
                        </div>
                        <div class="metric">
                            <strong>Total:</strong> ${stats.total_entries || 0}
                        </div>
                        <div class="metric">
                            <strong>Effizienz:</strong> ${(stats.memory_efficiency || 0).toFixed(1)}%
                        </div>
                    `;
                    document.getElementById('memoryData').innerHTML = html;
                } else {
                    document.getElementById('memoryData').innerHTML = '<span class="status status-error">Memory Stats nicht verfügbar</span>';
                }
            } catch (error) {
                document.getElementById('memoryData').innerHTML = '<span class="status status-error">Fehler beim Laden</span>';
            }
        }

        // Enter key support for login
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('password').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    login();
                }
            });
        });
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(len(html.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def handle_api_request(self):
        # Proxy to actual API
        self.send_response(404)
        self.end_headers()


def start_admin_panel():
    PORT = 8080

    with socketserver.TCPServer(("", PORT), AdminPanelHandler) as httpd:
        print(f"🚀 Admin-Panel gestartet auf http://localhost:{PORT}")
        print("📖 Öffne http://localhost:8080 in deinem Browser")
        print("🔐 Login: admin / admin123!")
        print("⏹️  Strg+C zum Beenden")

        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\\n👋 Admin-Panel beendet")


if __name__ == "__main__":
    start_admin_panel()
