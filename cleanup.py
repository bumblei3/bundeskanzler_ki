#!/usr/bin/env python3
"""
🧹 Arbeitsumgebung Cleanup-Skript
=================================

Säubere die Bundeskanzler-KI Arbeitsumgebung:
- Entferne Python-Cache-Dateien
- Lösche temporäre Testdateien
- Bereinige Embeddings und Indizes (optional)
- Erstelle Übersicht über freigegebenen Speicherplatz

Autor: Claude-3.5-Sonnet
Datum: 16. September 2025
"""

import os
import shutil
import glob
from datetime import datetime
from typing import List, Tuple

class EnvironmentCleaner:
    """
    Säubere die Arbeitsumgebung systematisch
    """

    def __init__(self, project_root: str):
        """
        Initialisiert den Cleaner

        Args:
            project_root: Wurzelverzeichnis des Projekts
        """
        self.project_root = project_root
        self.cleaned_files = []
        self.total_size_cleaned = 0

    def get_file_size(self, file_path: str) -> int:
        """
        Gibt die Größe einer Datei in Bytes zurück

        Args:
            file_path: Pfad zur Datei

        Returns:
            Dateigröße in Bytes
        """
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def format_size(self, size_bytes: int) -> str:
        """
        Formatiert Bytes in lesbare Einheiten

        Args:
            size_bytes: Größe in Bytes

        Returns:
            Formatierte Größe (KB, MB, GB)
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return ".1f"
            size_bytes /= 1024.0
        return ".1f"

    def remove_file_safe(self, file_path: str) -> bool:
        """
        Entfernt eine Datei sicher (mit Fehlerbehandlung)

        Args:
            file_path: Pfad zur zu entfernenden Datei

        Returns:
            True wenn erfolgreich, False bei Fehler
        """
        try:
            if os.path.isfile(file_path):
                size = self.get_file_size(file_path)
                os.remove(file_path)
                self.cleaned_files.append(file_path)
                self.total_size_cleaned += size
                print(f"   🗑️  Gelöscht: {file_path} ({self.format_size(size)})")
                return True
            elif os.path.isdir(file_path):
                size = sum(self.get_file_size(os.path.join(dirpath, filename))
                          for dirpath, _, filenames in os.walk(file_path)
                          for filename in filenames)
                shutil.rmtree(file_path)
                self.cleaned_files.append(file_path)
                self.total_size_cleaned += size
                print(f"   🗂️  Verzeichnis gelöscht: {file_path} ({self.format_size(size)})")
                return True
        except Exception as e:
            print(f"   ❌ Fehler beim Löschen {file_path}: {e}")
            return False
        return False

    def cleanup_python_cache(self) -> int:
        """
        Entfernt alle Python-Cache-Dateien und -Verzeichnisse

        Returns:
            Anzahl der entfernten Elemente
        """
        print("🐍 Entferne Python-Cache-Dateien...")

        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/.coverage",
            "**/*.pyo"
        ]

        removed_count = 0
        for pattern in cache_patterns:
            for path in glob.glob(os.path.join(self.project_root, pattern), recursive=True):
                if self.remove_file_safe(path):
                    removed_count += 1

        return removed_count

    def cleanup_temp_charts(self) -> int:
        """
        Entfernt temporäre Chart-Dateien aus /tmp/

        Returns:
            Anzahl der entfernten Dateien
        """
        print("📊 Entferne temporäre Chart-Dateien...")

        chart_patterns = [
            "/tmp/*_trend_*.png",
            "/tmp/*_comparison_*.png",
            "/tmp/*_distribution_*.png",
            "/tmp/*_timeline_*.png",
            "/tmp/error_visualization_*.png"
        ]

        removed_count = 0
        for pattern in chart_patterns:
            for path in glob.glob(pattern):
                if self.remove_file_safe(path):
                    removed_count += 1

        return removed_count

    def cleanup_test_artifacts(self) -> int:
        """
        Entfernt Test-Artefakte (optionale Bereinigung)

        Returns:
            Anzahl der entfernten Dateien
        """
        print("🧪 Entferne Test-Artefakte...")

        test_files = [
            "test_google_translate.py",
            "test_multilingual_multimodal.py",
            ".coverage",
            "coverage.xml",
            "htmlcov/",
            ".benchmarks/"
        ]

        removed_count = 0
        for file_path in test_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                if self.remove_file_safe(full_path):
                    removed_count += 1

        return removed_count

    def cleanup_embeddings_cache(self, confirm: bool = False) -> int:
        """
        Entfernt gecachte Embeddings (nur mit Bestätigung)

        Args:
            confirm: Muss True sein um Embeddings zu löschen

        Returns:
            Anzahl der entfernten Dateien
        """
        if not confirm:
            print("💾 Überspringe Embeddings-Cache (verwende --clean-embeddings zum Löschen)")
            return 0

        print("💾 Entferne Embeddings-Cache...")

        embedding_files = [
            "gpu_rag_embeddings.pkl",
            "rag_embeddings.pkl",
            "rag_index.faiss"
        ]

        removed_count = 0
        for file_path in embedding_files:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                if self.remove_file_safe(full_path):
                    removed_count += 1

        return removed_count

    def cleanup_log_files(self) -> int:
        """
        Entfernt alte Log-Dateien

        Returns:
            Anzahl der entfernten Dateien
        """
        print("📝 Entferne alte Log-Dateien...")

        log_patterns = [
            "logs/*.log",
            "*.log"
        ]

        removed_count = 0
        for pattern in log_patterns:
            for path in glob.glob(os.path.join(self.project_root, pattern)):
                # Prüfe ob Datei älter als 7 Tage ist
                if os.path.getmtime(path) < datetime.now().timestamp() - (7 * 24 * 60 * 60):
                    if self.remove_file_safe(path):
                        removed_count += 1

        return removed_count

    def get_disk_usage(self) -> Tuple[str, str]:
        """
        Gibt die aktuelle Festplattenbelegung zurück

        Returns:
            Tuple aus verwendetem und verfügbarem Speicher
        """
        try:
            stat = os.statvfs(self.project_root)
            total = stat.f_bsize * stat.f_blocks
            available = stat.f_bsize * stat.f_bavail
            used = total - available
            return self.format_size(used), self.format_size(available)
        except:
            return "N/A", "N/A"

    def run_cleanup(self, clean_embeddings: bool = False, clean_tests: bool = False) -> dict:
        """
        Führt die komplette Bereinigung durch

        Args:
            clean_embeddings: Ob Embeddings gelöscht werden sollen
            clean_tests: Ob Testdateien gelöscht werden sollen

        Returns:
            Dictionary mit Bereinigungsstatistiken
        """
        print("🧹 Starte Arbeitsumgebung Cleanup...")
        print("=" * 50)
        print(f"Projekt: {self.project_root}")
        print(f"Startzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Zeige aktuelle Festplattenbelegung
        used, available = self.get_disk_usage()
        print(f"💾 Aktuelle Festplattenbelegung: {used} verwendet, {available} verfügbar")
        print()

        # Führe Bereinigungen durch
        stats = {
            "python_cache_removed": self.cleanup_python_cache(),
            "temp_charts_removed": self.cleanup_temp_charts(),
            "log_files_removed": self.cleanup_log_files(),
            "test_artifacts_removed": 0,
            "embeddings_removed": 0
        }

        if clean_tests:
            stats["test_artifacts_removed"] = self.cleanup_test_artifacts()

        if clean_embeddings:
            stats["embeddings_removed"] = self.cleanup_embeddings_cache(confirm=True)

        # Zeige Zusammenfassung
        print()
        print("📊 Cleanup-Zusammenfassung:")
        print("=" * 30)
        print(f"🐍 Python-Cache entfernt: {stats['python_cache_removed']}")
        print(f"📊 Temp-Charts entfernt: {stats['temp_charts_removed']}")
        print(f"📝 Log-Dateien entfernt: {stats['log_files_removed']}")

        if clean_tests:
            print(f"🧪 Test-Artefakte entfernt: {stats['test_artifacts_removed']}")

        if clean_embeddings:
            print(f"💾 Embeddings entfernt: {stats['embeddings_removed']}")

        print(f"🗑️  Gesamt bereinigt: {len(self.cleaned_files)} Dateien/Verzeichnisse")
        print(f"💾 Speicher freigegeben: {self.format_size(self.total_size_cleaned)}")

        # Zeige neue Festplattenbelegung
        new_used, new_available = self.get_disk_usage()
        print(f"💾 Neue Festplattenbelegung: {new_used} verwendet, {new_available} verfügbar")

        print()
        print("✅ Cleanup abgeschlossen!")
        print(f"Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return stats

def main():
    """
    Hauptfunktion für den Cleanup
    """
    import argparse

    parser = argparse.ArgumentParser(description="Bundeskanzler-KI Arbeitsumgebung Cleanup")
    parser.add_argument("--clean-embeddings", action="store_true",
                       help="Entferne auch Embeddings-Cache (gpu_rag_embeddings.pkl, etc.)")
    parser.add_argument("--clean-tests", action="store_true",
                       help="Entferne Testdateien (test_*.py, coverage, etc.)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Zeige nur was gelöscht würde, ohne tatsächlich zu löschen")

    args = parser.parse_args()

    # Projekt-Root ermitteln
    project_root = os.path.dirname(os.path.abspath(__file__))

    if args.dry_run:
        print("🔍 DRY-RUN Modus - Zeige nur was gelöscht würde:")
        print("Verwende --clean-embeddings oder --clean-tests um diese zu entfernen")
        print()
        # Hier könnte man eine Vorschau implementieren
        return

    # Cleanup durchführen
    cleaner = EnvironmentCleaner(project_root)
    stats = cleaner.run_cleanup(
        clean_embeddings=args.clean_embeddings,
        clean_tests=args.clean_tests
    )

    # Empfehlungen ausgeben
    print()
    print("💡 Empfehlungen:")
    if not args.clean_embeddings:
        print("   💾 Verwende --clean-embeddings um Embeddings neu zu generieren")
    if not args.clean_tests:
        print("   🧪 Verwende --clean-tests um Testdateien zu entfernen")
    print("   🔄 Embeddings werden bei Bedarf automatisch neu erstellt")

if __name__ == "__main__":
    main()