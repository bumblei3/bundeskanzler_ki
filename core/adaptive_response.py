#!/usr/bin/env python3
"""
Adaptive Response Manager für Bundeskanzler-KI
Passt Antworten an Benutzerfeedback an
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np


class AdaptiveResponseManager:
    """Manager für adaptive Antworten basierend auf Feedback"""
    
    def __init__(self, feedback_file: str = "data/feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback()
        
    def _load_feedback(self) -> Dict[str, Any]:
        """Lädt Feedback-Daten"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"responses": {}, "patterns": {}}
    
    def _save_feedback(self):
        """Speichert Feedback-Daten"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
    
    def record_feedback(self, question: str, answer: str, rating: float, user_id: Optional[str] = None):
        """Zeichnet Feedback auf"""
        key = self._generate_key(question)
        
        if key not in self.feedback_data["responses"]:
            self.feedback_data["responses"][key] = {
                "question": question,
                "answers": [],
                "ratings": [],
                "timestamps": []
            }
        
        self.feedback_data["responses"][key]["answers"].append(answer)
        self.feedback_data["responses"][key]["ratings"].append(rating)
        self.feedback_data["responses"][key]["timestamps"].append(datetime.now().isoformat())
        
        self._save_feedback()
    
    def get_adaptive_response(self, question: str) -> Optional[str]:
        """Gibt adaptive Antwort zurück"""
        key = self._generate_key(question)
        
        if key in self.feedback_data["responses"]:
            responses = self.feedback_data["responses"][key]
            if responses["ratings"]:
                # Bester Rating finden
                best_idx = np.argmax(responses["ratings"])
                return responses["answers"][best_idx]
        
        return None
    
    def _generate_key(self, question: str) -> str:
        """Generiert Schlüssel für Frage"""
        # Einfache Normalisierung
        return question.lower().strip()[:100]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Gibt Statistiken zurück"""
        total_responses = len(self.feedback_data["responses"])
        total_ratings = sum(len(r["ratings"]) for r in self.feedback_data["responses"].values())
        avg_rating = 0
        
        if total_ratings > 0:
            all_ratings = []
            for r in self.feedback_data["responses"].values():
                all_ratings.extend(r["ratings"])
            avg_rating = np.mean(all_ratings)
        
        return {
            "total_responses": total_responses,
            "total_ratings": total_ratings,
            "average_rating": float(avg_rating)
        }


# Globale Instanz
adaptive_response_manager = AdaptiveResponseManager()
