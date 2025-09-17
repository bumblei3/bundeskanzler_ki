#!/usr/bin/env python3
"""
Pytest-Konfiguration für sauberes Thread-Management
"""

import pytest
import threading
import time
import atexit


@pytest.fixture(scope="session", autouse=True)
def cleanup_threads():
    """Stellt sicher, dass alle Threads ordnungsgemäß beendet werden"""
    yield

    # Sammle alle aktiven Threads
    main_thread = threading.main_thread()
    active_threads = []

    for thread in threading.enumerate():
        if thread is not main_thread and thread.is_alive():
            active_threads.append(thread)

    if active_threads:
        print(f"\n🧹 Beende {len(active_threads)} aktive Threads...")

        # Beende alle Threads
        for thread in active_threads:
            if hasattr(thread, 'daemon') and thread.daemon:
                # Daemon-Threads werden automatisch beendet
                continue
            else:
                # Versuche nicht-daemon Threads zu beenden
                try:
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        print(f"⚠️ Thread {thread.name} konnte nicht beendet werden")
                except Exception as e:
                    print(f"⚠️ Fehler beim Beenden von Thread {thread.name}: {e}")

    # Force Garbage Collection
    import gc
    gc.collect()


# Zusätzliche Cleanup-Funktion beim Programmende
def emergency_cleanup():
    """Notfall-Cleanup für alle verbleibenden Threads"""
    try:
        main_thread = threading.main_thread()
        for thread in threading.enumerate():
            if thread is not main_thread and thread.is_alive():
                if not hasattr(thread, 'daemon') or not thread.daemon:
                    try:
                        thread.join(timeout=0.1)
                    except:
                        pass
    except:
        pass


atexit.register(emergency_cleanup)