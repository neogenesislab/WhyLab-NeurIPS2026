import json
import random
import os
from typing import Dict, Any, Optional

class LabDirector:
    def __init__(self, knowledge_path: str = "data/grand_challenges.json"):
        self.knowledge_path = knowledge_path
        self.challenges = []
        self.current_agenda: Optional[Dict[str, Any]] = None
        self._load_challenges()

    def _load_challenges(self):
        """Load grand challenges from JSON file."""
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                    self.challenges = json.load(f)
            except Exception as e:
                print(f"[Director] Failed to load challenges: {e}")
                self.challenges = []
        else:
            print(f"[Director] Challenge DB not found at {self.knowledge_path}")

    def set_agenda(self, agenda_id: str = None) -> Dict[str, Any]:
        """Set a specific agenda or pick a random one."""
        if not self.challenges:
            return {"error": "No challenges available"}

        if agenda_id:
            target = next((c for c in self.challenges if c["id"] == agenda_id), None)
            if target:
                self.current_agenda = target
                print(f"[Director] Agenda set to: {target['title']}")
                return target

        # Random pick if no ID provided or not found
        self.current_agenda = random.choice(self.challenges)
        print(f"[Director] New Agenda Selected: {self.current_agenda['title']}")
        return self.current_agenda

    def get_current_agenda(self) -> Dict[str, Any]:
        """Return the current active agenda."""
        if not self.current_agenda:
            return self.set_agenda() # Auto-set if empty
        return self.current_agenda

    def announce_directive(self) -> str:
        """Generate a directive message for the team."""
        if not self.current_agenda:
            self.set_agenda()
        
        agenda = self.current_agenda
        return (
            f"📢 **Attention Team**: The current research focus is '{agenda['title']}'.\n"
            f"🎯 **Goal**: {agenda['description']}\n"
            f"⚠️ **Impact**: {agenda['impact']} | **Difficulty**: {agenda['difficulty']}\n"
            "Theorist, please initialize hypotheses. Engineer, prepare simulations."
        )
