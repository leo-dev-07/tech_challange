"""Feedback system for learning from user interactions.

Captures user feedback on agent responses and uses it to improve future responses.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


class FeedbackStore:
    """Manages feedback collection, storage, and analysis."""
    
    def __init__(self, feedback_dir: str = "feedback"):
        """Initialize feedback store.
        
        Args:
            feedback_dir: Directory to store feedback files
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        self.feedback_file = self.feedback_dir / "interactions.jsonl"
        self.summary_file = self.feedback_dir / "summary.json"
    
    def add_feedback(
        self,
        query: str,
        response: str,
        rating: str,  # "good", "bad", "neutral"
        correction: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Record feedback for a query-response pair.
        
        Args:
            query: The user's original query
            response: The agent's response
            rating: User's rating: "good", "bad", or "neutral"
            correction: Optional corrected/improved response from user
            tags: Optional tags to categorize the feedback (e.g., ["too_long", "wrong_data"])
        """
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "rating": rating,
            "correction": correction,
            "tags": tags or []
        }
        
        # Append to feedback file
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        # Update summary
        self._update_summary(rating, tags)
    
    def _update_summary(self, rating: str, tags: Optional[List[str]] = None) -> None:
        """Update feedback summary statistics."""
        summary = self._load_summary()
        
        # Update rating counts
        if rating not in summary["ratings"]:
            summary["ratings"][rating] = 0
        summary["ratings"][rating] += 1
        
        # Update tag counts
        if tags:
            for tag in tags:
                if tag not in summary["tags"]:
                    summary["tags"][tag] = 0
                summary["tags"][tag] += 1
        
        # Update total interactions
        summary["total_interactions"] += 1
        summary["last_updated"] = datetime.now().isoformat()
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    def _load_summary(self) -> Dict[str, Any]:
        """Load or create feedback summary."""
        if self.summary_file.exists():
            with open(self.summary_file, "r") as f:
                return json.load(f)
        return {
            "total_interactions": 0,
            "ratings": {},
            "tags": {},
            "last_updated": None
        }
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return self._load_summary()
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent feedback entries."""
        if not self.feedback_file.exists():
            return []
        
        entries = []
        with open(self.feedback_file, "r") as f:
            for line in f:
                entries.append(json.loads(line))
        
        return entries[-limit:]
    
    def get_bad_responses(self) -> List[Dict[str, Any]]:
        """Get all negative feedback entries."""
        if not self.feedback_file.exists():
            return []
        
        bad_entries = []
        with open(self.feedback_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                if entry["rating"] == "bad":
                    bad_entries.append(entry)
        
        return bad_entries
    
    def get_common_issues(self, top_n: int = 5) -> List[tuple]:
        """Get most common feedback tags (issues users reported)."""
        stats = self.get_feedback_stats()
        tags = stats.get("tags", {})
        
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:top_n]
    
    def should_avoid_pattern(self, query: str) -> bool:
        """Check if this type of query has received bad feedback."""
        bad_responses = self.get_bad_responses()
        
        # Simple keyword matching to detect patterns
        query_lower = query.lower()
        for entry in bad_responses:
            if query_lower in entry["query"].lower() or entry["query"].lower() in query_lower:
                return True
        
        return False
    
    def get_corrections_for_pattern(self, pattern: str) -> List[str]:
        """Get user corrections for similar queries."""
        corrections = []
        bad_responses = self.get_bad_responses()
        
        pattern_lower = pattern.lower()
        for entry in bad_responses:
            if pattern_lower in entry["query"].lower() and entry.get("correction"):
                corrections.append(entry["correction"])
        
        return corrections
