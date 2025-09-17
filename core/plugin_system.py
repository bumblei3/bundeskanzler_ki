#!/usr/bin/env python3
"""
Plugin-System für Bundeskanzler KI
Modulares System für Erweiterungen und benutzerdefinierte Funktionen
"""

import abc
import importlib
import inspect
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PluginMetadata:
    """Metadaten für ein Plugin"""
    name: str
    version: str
    description: str
    author: str
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    repository: Optional[str] = None
    min_python_version: str = "3.8"
    max_python_version: Optional[str] = None

@dataclass
class PluginConfig:
    """Konfiguration für ein Plugin"""
    enabled: bool = True
    priority: int = 100
    settings: Dict[str, Any] = field(default_factory=dict)
    auto_start: bool = True

class PluginError(Exception):
    """Basisklasse für Plugin-Fehler"""
    pass

class PluginLoadError(PluginError):
    """Fehler beim Laden eines Plugins"""
    pass

class PluginExecutionError(PluginError):
    """Fehler bei der Ausführung eines Plugins"""
    pass

class PluginSecurityError(PluginError):
    """Sicherheitsfehler bei Plugin-Ausführung"""
    pass

class BasePlugin(abc.ABC):
    """
    Basis-Klasse für alle Plugins

    Alle Plugins müssen von dieser Klasse erben und die abstrakten Methoden implementieren.
    """

    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._config: PluginConfig = PluginConfig()
        self._metadata: Optional[PluginMetadata] = None
        self._initialized = False

    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Gibt die Metadaten des Plugins zurück"""
        pass

    @abc.abstractmethod
    def initialize(self) -> None:
        """
        Initialisiert das Plugin

        Diese Methode wird einmal beim Laden des Plugins aufgerufen.
        Hier sollten Ressourcen initialisiert und Abhängigkeiten geprüft werden.
        """
        pass

    @abc.abstractmethod
    def shutdown(self) -> None:
        """
        Beendet das Plugin

        Diese Methode wird beim Herunterfahren des Systems aufgerufen.
        Hier sollten Ressourcen freigegeben werden.
        """
        pass

    def get_config(self) -> PluginConfig:
        """Gibt die aktuelle Plugin-Konfiguration zurück"""
        return self._config

    def update_config(self, config: Dict[str, Any]) -> None:
        """Aktualisiert die Plugin-Konfiguration"""
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self._config.settings[key] = value

    def is_enabled(self) -> bool:
        """Prüft, ob das Plugin aktiviert ist"""
        return self._config.enabled

    def enable(self) -> None:
        """Aktiviert das Plugin"""
        self._config.enabled = True
        self.logger.info(f"Plugin {self.metadata.name} aktiviert")

    def disable(self) -> None:
        """Deaktiviert das Plugin"""
        self._config.enabled = False
        self.logger.info(f"Plugin {self.metadata.name} deaktiviert")

    def execute_safe(self, method_name: str, *args, **kwargs) -> Any:
        """
        Führt eine Plugin-Methode sicher aus mit Fehlerbehandlung

        Args:
            method_name: Name der auszuführenden Methode
            *args: Positionsargumente für die Methode
            **kwargs: Schlüsselwortargumente für die Methode

        Returns:
            Das Ergebnis der Methoden-Ausführung

        Raises:
            PluginExecutionError: Bei Fehlern während der Ausführung
        """
        if not self.is_enabled():
            raise PluginExecutionError(f"Plugin {self.metadata.name} ist deaktiviert")

        try:
            method = getattr(self, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Fehler bei Ausführung von {method_name}: {e}")
            raise PluginExecutionError(f"Plugin {method_name} fehlgeschlagen: {e}") from e

class TextProcessingPlugin(BasePlugin):
    """
    Basis-Klasse für Text-Verarbeitungs-Plugins

    Diese Plugins können Text vorverarbeiten, analysieren oder modifizieren.
    """

    @abc.abstractmethod
    def process_text(self, text: str, **kwargs) -> str:
        """
        Verarbeitet Text-Eingaben

        Args:
            text: Der zu verarbeitende Text
            **kwargs: Zusätzliche Parameter

        Returns:
            Der verarbeitete Text
        """
        pass

class ImageProcessingPlugin(BasePlugin):
    """
    Basis-Klasse für Bild-Verarbeitungs-Plugins

    Diese Plugins können Bilder analysieren, modifizieren oder generieren.
    """

    @abc.abstractmethod
    def process_image(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Verarbeitet Bild-Eingaben

        Args:
            image_path: Pfad zur Bilddatei
            **kwargs: Zusätzliche Parameter

        Returns:
            Dictionary mit Analyse-Ergebnissen
        """
        pass

class AudioProcessingPlugin(BasePlugin):
    """
    Basis-Klasse für Audio-Verarbeitungs-Plugins

    Diese Plugins können Audio transkribieren, analysieren oder generieren.
    """

    @abc.abstractmethod
    def process_audio(self, audio_path: str, **kwargs) -> str:
        """
        Verarbeitet Audio-Eingaben

        Args:
            audio_path: Pfad zur Audiodatei
            **kwargs: Zusätzliche Parameter

        Returns:
            Transkribierter Text oder Analyse-Ergebnis
        """
        pass

class HookPlugin(BasePlugin):
    """
    Basis-Klasse für Hook-Plugins

    Diese Plugins können sich in verschiedene Systemereignisse einhängen.
    """

    def on_request_start(self, request_data: Dict[str, Any]) -> None:
        """Wird aufgerufen, wenn eine neue Anfrage startet"""
        pass

    def on_request_end(self, request_data: Dict[str, Any], response: Any) -> None:
        """Wird aufgerufen, wenn eine Anfrage beendet wird"""
        pass

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Wird aufgerufen, wenn ein Fehler auftritt"""
        pass

    def on_system_start(self) -> None:
        """Wird aufgerufen, wenn das System startet"""
        pass

    def on_system_shutdown(self) -> None:
        """Wird aufgerufen, wenn das System heruntergefahren wird"""
        pass

class PluginManager:
    """
    Zentraler Plugin-Manager für das Laden und Verwalten von Plugins

    Verwaltet den Plugin-Lebenszyklus, Konfiguration und Sicherheit.
    """

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or [
            Path(__file__).parent / "plugins",
            Path.home() / ".bundeskanzler" / "plugins",
            "/opt/bundeskanzler/plugins"
        ]

        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        self.hooks: Dict[str, List[BasePlugin]] = {}

        # Thread-Sicherheit
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Plugin-Registry für Sicherheit
        self.allowed_plugins: List[str] = []
        self.blocked_plugins: List[str] = []

        logger.info("Plugin-Manager initialisiert")

    def discover_plugins(self) -> List[str]:
        """
        Entdeckt verfügbare Plugins in den Plugin-Verzeichnissen

        Returns:
            Liste der gefundenen Plugin-Namen
        """
        discovered_plugins = []

        for plugin_dir in self.plugin_dirs:
            if not Path(plugin_dir).exists():
                continue

            for item in Path(plugin_dir).iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    plugin_name = item.name
                    if plugin_name not in discovered_plugins:
                        discovered_plugins.append(plugin_name)

        logger.info(f"{len(discovered_plugins)} Plugins entdeckt")
        return discovered_plugins

    def load_plugin(self, plugin_name: str, config: Optional[PluginConfig] = None) -> bool:
        """
        Lädt ein einzelnes Plugin

        Args:
            plugin_name: Name des zu ladenden Plugins
            config: Optionale Plugin-Konfiguration

        Returns:
            True wenn das Plugin erfolgreich geladen wurde
        """
        with self._lock:
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} bereits geladen")
                return True

            if plugin_name in self.blocked_plugins:
                logger.warning(f"Plugin {plugin_name} ist blockiert")
                return False

            try:
                # Plugin-Modul importieren
                plugin_module = self._import_plugin_module(plugin_name)
                if not plugin_module:
                    return False

                # Plugin-Klasse finden
                plugin_class = self._find_plugin_class(plugin_module)
                if not plugin_class:
                    return False

                # Plugin instanziieren
                plugin_instance = plugin_class(self)

                # Konfiguration anwenden
                if config:
                    plugin_instance.update_config(config.__dict__)

                # Plugin initialisieren
                plugin_instance.initialize()

                # Plugin registrieren
                self.plugins[plugin_name] = plugin_instance
                self.plugin_configs[plugin_name] = plugin_instance.get_config()

                # Hooks registrieren
                if isinstance(plugin_instance, HookPlugin):
                    self._register_hooks(plugin_instance)

                logger.info(f"Plugin {plugin_name} erfolgreich geladen")
                return True

            except Exception as e:
                logger.error(f"Fehler beim Laden von Plugin {plugin_name}: {e}")
                return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Entlädt ein Plugin

        Args:
            plugin_name: Name des zu entladenden Plugins

        Returns:
            True wenn das Plugin erfolgreich entladen wurde
        """
        with self._lock:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} nicht geladen")
                return False

            try:
                plugin = self.plugins[plugin_name]

                # Plugin herunterfahren
                plugin.shutdown()

                # Hooks entfernen
                if isinstance(plugin, HookPlugin):
                    self._unregister_hooks(plugin)

                # Plugin entfernen
                del self.plugins[plugin_name]
                del self.plugin_configs[plugin_name]

                logger.info(f"Plugin {plugin_name} erfolgreich entladen")
                return True

            except Exception as e:
                logger.error(f"Fehler beim Entladen von Plugin {plugin_name}: {e}")
                return False

    def load_all_plugins(self) -> int:
        """
        Lädt alle verfügbaren Plugins

        Returns:
            Anzahl der erfolgreich geladenen Plugins
        """
        discovered_plugins = self.discover_plugins()
        loaded_count = 0

        for plugin_name in discovered_plugins:
            if self.load_plugin(plugin_name):
                loaded_count += 1

        logger.info(f"{loaded_count} von {len(discovered_plugins)} Plugins geladen")
        return loaded_count

    def unload_all_plugins(self) -> int:
        """
        Entlädt alle geladenen Plugins

        Returns:
            Anzahl der erfolgreich entladenen Plugins
        """
        plugin_names = list(self.plugins.keys())
        unloaded_count = 0

        for plugin_name in plugin_names:
            if self.unload_plugin(plugin_name):
                unloaded_count += 1

        logger.info(f"{unloaded_count} Plugins entladen")
        return unloaded_count

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Gibt eine Plugin-Instanz zurück

        Args:
            plugin_name: Name des Plugins

        Returns:
            Plugin-Instanz oder None wenn nicht gefunden
        """
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> List[BasePlugin]:
        """
        Gibt alle Plugins eines bestimmten Typs zurück

        Args:
            plugin_type: Plugin-Basisklasse

        Returns:
            Liste der Plugins des angegebenen Typs
        """
        return [plugin for plugin in self.plugins.values() if isinstance(plugin, plugin_type)]

    def execute_hook(self, hook_name: str, *args, **kwargs) -> None:
        """
        Führt alle Plugins aus, die einen bestimmten Hook implementieren

        Args:
            hook_name: Name des Hooks
            *args: Argumente für den Hook
            **kwargs: Schlüsselwortargumente für den Hook
        """
        if hook_name not in self.hooks:
            return

        for plugin in self.hooks[hook_name]:
            if plugin.is_enabled():
                try:
                    hook_method = getattr(plugin, hook_name)
                    self._executor.submit(hook_method, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Fehler beim Ausführen von Hook {hook_name} in Plugin {plugin.metadata.name}: {e}")

    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt Informationen über alle geladenen Plugins zurück

        Returns:
            Dictionary mit Plugin-Informationen
        """
        info = {}
        for name, plugin in self.plugins.items():
            info[name] = {
                "metadata": plugin.metadata.__dict__,
                "config": plugin.get_config().__dict__,
                "enabled": plugin.is_enabled(),
                "type": plugin.__class__.__name__
            }
        return info

    def _import_plugin_module(self, plugin_name: str) -> Optional[Any]:
        """Importiert ein Plugin-Modul"""
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir) / plugin_name
            if plugin_path.exists():
                try:
                    # Plugin-Verzeichnis zum Python-Pfad hinzufügen
                    if str(plugin_path) not in sys.path:
                        sys.path.insert(0, str(plugin_path))

                    # Plugin importieren
                    return importlib.import_module(plugin_name)
                except Exception as e:
                    logger.error(f"Fehler beim Importieren von Plugin {plugin_name}: {e}")
                    return None
        return None

    def _find_plugin_class(self, plugin_module: Any) -> Optional[Type[BasePlugin]]:
        """Findet die Plugin-Klasse in einem Modul"""
        for name, obj in inspect.getmembers(plugin_module):
            if (inspect.isclass(obj) and
                issubclass(obj, BasePlugin) and
                obj != BasePlugin and
                obj != TextProcessingPlugin and
                obj != ImageProcessingPlugin and
                obj != AudioProcessingPlugin and
                obj != HookPlugin):
                return obj
        return None

    def _register_hooks(self, plugin: HookPlugin) -> None:
        """Registriert Hook-Methoden eines Plugins"""
        hook_methods = [
            'on_request_start', 'on_request_end', 'on_error',
            'on_system_start', 'on_system_shutdown'
        ]

        for hook_name in hook_methods:
            if hasattr(plugin, hook_name):
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                self.hooks[hook_name].append(plugin)

    def _unregister_hooks(self, plugin: HookPlugin) -> None:
        """Entfernt Hook-Methoden eines Plugins"""
        for hook_name, plugins in self.hooks.items():
            if plugin in plugins:
                plugins.remove(plugin)

# Globale Plugin-Manager Instanz
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """
    Gibt die globale Plugin-Manager Instanz zurück

    Returns:
        PluginManager: Globale Plugin-Manager Instanz
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def initialize_plugin_system() -> PluginManager:
    """
    Initialisiert das Plugin-System

    Returns:
        PluginManager: Initialisierter Plugin-Manager
    """
    manager = get_plugin_manager()
    manager.load_all_plugins()
    manager.execute_hook("on_system_start")
    return manager

def shutdown_plugin_system() -> None:
    """
    Beendet das Plugin-System
    """
    global _plugin_manager
    if _plugin_manager:
        _plugin_manager.execute_hook("on_system_shutdown")
        _plugin_manager.unload_all_plugins()
        _plugin_manager = None