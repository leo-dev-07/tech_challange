"""Data loader module for the query agent.

Loads and indexes generated demo data (companies, contacts, deals, etc.)
for quick retrieval during agent queries.
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path


class DataStore:
    """In-memory store for all generated demo data."""

    def __init__(self, data_path: str = "output"):
        """Load all JSON files from the data directory.

        Args:
            data_path: directory containing companies.json, contacts.json, etc.
        """
        self.data_path = data_path
        self.companies: List[Dict] = []
        self.contacts: List[Dict] = []
        self.sales_reps: List[Dict] = []
        self.deals: List[Dict] = []
        self.emails: List[Dict] = []
        self.meetings: List[Dict] = []

        # Build indices for fast lookup
        self.companies_by_id: Dict[str, Dict] = {}
        self.contacts_by_id: Dict[str, Dict] = {}
        self.deals_by_id: Dict[str, Dict] = {}
        self.reps_by_id: Dict[str, Dict] = {}

        self._load_data()

    def _load_data(self):
        """Load all JSON files from the data directory."""
        files = {
            "companies.json": "companies",
            "contacts.json": "contacts",
            "sales_reps.json": "sales_reps",
            "deals.json": "deals",
            "emails.json": "emails",
            "meetings.json": "meetings",
        }

        for filename, attr in files.items():
            path = os.path.join(self.data_path, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    setattr(self, attr, data)
                    print(f"Loaded {len(data)} records from {filename}")

        # Build indices
        self.companies_by_id = {c["company_id"]: c for c in self.companies}
        self.contacts_by_id = {c["contact_id"]: c for c in self.contacts}
        self.deals_by_id = {d["deal_id"]: d for d in self.deals}
        self.reps_by_id = {r["rep_id"]: r for r in self.sales_reps}

    def get_company(self, company_id: str) -> Dict:
        """Retrieve a company by ID."""
        return self.companies_by_id.get(company_id)

    def get_contact(self, contact_id: str) -> Dict:
        """Retrieve a contact by ID."""
        return self.contacts_by_id.get(contact_id)

    def get_deal(self, deal_id: str) -> Dict:
        """Retrieve a deal by ID."""
        return self.deals_by_id.get(deal_id)

    def get_rep(self, rep_id: str) -> Dict:
        """Retrieve a sales rep by ID."""
        return self.reps_by_id.get(rep_id)

    def search_companies(self, query: str) -> List[Dict]:
        """Simple text search in company names and industries."""
        q = query.lower()
        return [c for c in self.companies if q in c.get("name", "").lower() or q in c.get("industry", "").lower()]

    def search_contacts(self, query: str) -> List[Dict]:
        """Simple text search in contact names and emails."""
        q = query.lower()
        return [c for c in self.contacts if q in c.get("full_name", "").lower() or q in c.get("email", "").lower()]

    def get_deals_by_company(self, company_id: str) -> List[Dict]:
        """Get all deals for a specific company."""
        return [d for d in self.deals if d["company_id"] == company_id]

    def get_contacts_by_company(self, company_id: str) -> List[Dict]:
        """Get all contacts for a specific company."""
        return [c for c in self.contacts if c["company_id"] == company_id]

    def get_emails_by_deal(self, deal_id: str) -> List[Dict]:
        """Get all emails for a specific deal."""
        return [e for e in self.emails if e["deal_id"] == deal_id]

    def get_meetings_by_deal(self, deal_id: str) -> List[Dict]:
        """Get all meetings for a specific deal."""
        return [m for m in self.meetings if m["deal_id"] == deal_id]

    def get_summary(self) -> Dict[str, Any]:
        """Return a high-level summary of the dataset."""
        return {
            "total_companies": len(self.companies),
            "total_contacts": len(self.contacts),
            "total_reps": len(self.sales_reps),
            "total_deals": len(self.deals),
            "total_emails": len(self.emails),
            "total_meetings": len(self.meetings),
            "industries": list(set(c.get("industry") for c in self.companies if "industry" in c)),
            "deal_stages": list(set(d.get("stage") for d in self.deals if "stage" in d)),
            "rep_tiers": list(set(r.get("tier") for r in self.sales_reps if "tier" in r)),
        }
