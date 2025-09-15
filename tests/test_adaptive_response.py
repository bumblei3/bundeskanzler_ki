import pytest
from adaptive_response import AdaptiveResponseManager

def test_get_complexity_params_basic():
    arm = AdaptiveResponseManager()
    user_id = "user1"
    context = {
        "current_topic": "Klimaschutz",
        "recent_messages": ["A", "B"],
        "technical_terms": 2,
        "question_complexity": 0.7
    }
    params = arm.get_complexity_params(user_id, context)
    assert 0.1 <= params["target_complexity"] <= 1.0
    assert params["min_length"] >= arm.min_tokens
    assert params["max_length"] <= arm.max_tokens
    assert 0.1 <= params["vocab_level"] <= 1.0
    assert "max_clause_depth" in params["structure_complexity"]

def test_update_user_profile_increases_complexity():
    arm = AdaptiveResponseManager()
    user_id = "user2"
    # Initialwert
    base_before = arm._get_or_create_profile(user_id)["base_complexity"]
    # Erfolgreiche Interaktion
    arm.update_user_profile(user_id, {"success": 0.8, "comprehension": 0.8})
    base_after = arm._get_or_create_profile(user_id)["base_complexity"]
    assert base_after >= base_before

def test_update_user_profile_decreases_complexity():
    arm = AdaptiveResponseManager()
    user_id = "user3"
    arm._get_or_create_profile(user_id)["base_complexity"] = 0.8
    arm.update_user_profile(user_id, {"success": 0.2, "comprehension": 0.5})
    base_after = arm._get_or_create_profile(user_id)["base_complexity"]
    assert base_after < 0.8

def test_history_limit():
    arm = AdaptiveResponseManager()
    user_id = "user4"
    for i in range(120):
        arm.update_user_profile(user_id, {"success": 0.5, "comprehension": 0.5})
    history = arm._get_or_create_profile(user_id)["history"]
    assert len(history) <= 100
