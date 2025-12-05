from __future__ import annotations
"""Core generator module.

This module defines simple dataclasses for the domain entities (Company, Contact,
SalesRep, Deal, Email, Meeting) and a `Generator` class that creates deterministic
demo data for each entity type. Faker and Python's random module are used for
realistic values; a `seed` can be provided to reproduce runs.
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4
from faker import Faker

fake = Faker()

# Small helper for unique IDs used across all models
IDS = lambda: str(uuid4())

@dataclass
class Company:
    """Represents a company (top-level account).

    Fields mirror the required output schema (company_id, industry, employee_count,
    revenue, and a short `context_note` describing a business pain/theme).
    """
    company_id: str
    name: str
    industry: str
    employee_count: int
    growth_stage: str
    hq_region: str
    annual_revenue_usd: int
    context_note: str

@dataclass
class Contact:
    """Represents a person at a company (prospect contact).

    `relationship_note` is a short textual reference to prior interactions or
    role relevance used for richer demo output.
    """
    contact_id: str
    company_id: str
    full_name: str
    title: str
    seniority: str
    email: str
    relationship_note: str

@dataclass
class SalesRep:
    """Internal sales representative assigned to deals.

    `tier` drives a simple target range stored in `quarter_deals_closed_target`.
    """
    rep_id: str
    full_name: str
    tier: str
    quarter_deals_closed_target: int

@dataclass
class Deal:
    """Represents a sales opportunity for a company.

    Dates are ISO8601 strings. `expected_close_at` is only set for active deals;
    `closed_at` and `loss_reason` are populated for closed deals.
    """
    deal_id: str
    company_id: str
    primary_contact_id: str
    rep_id: str
    stage: str
    health: str
    value_usd: int
    opened_at: str
    expected_close_at: Optional[str]
    closed_at: Optional[str]
    loss_reason: Optional[str]

@dataclass
class Email:
    """Represents an email message in a threaded conversation.

    `direction` is either 'outbound' (rep -> prospect) or 'inbound' (prospect -> rep).
    `reply_latency_hours` is populated for inbound messages to model response times.
    """
    email_id: str
    deal_id: str
    thread_id: str
    direction: str
    sender_id: str
    recipient_ids: List[str]
    timestamp: str
    subject: str
    body: str
    sequence_index: int
    sentiment: str
    trackers: List[str]
    reply_latency_hours: Optional[int]
    in_reply_to_email_id: Optional[str]
    outcome: str

@dataclass
class Meeting:
    """Represents scheduled/actual meetings attached to a deal.

    Meetings include attendees, a short notes field and an outcome. Sentiment
    and trackers provide behavioral context for the demo dataset.
    """
    meeting_id: str
    deal_id: str
    title: str
    stage_at_time: str
    scheduled_start: str
    scheduled_end: str
    actual_start: Optional[str]
    actual_end: Optional[str]
    attendee_ids: List[str]
    outcome: str
    notes: str
    sentiment: str
    trackers: List[str]
    follow_up_action: str


class Generator:
    # Constants used by the generator to model stages, health and growth categories
    STAGES = ["Prospecting", "Qualified", "Demo", "Proposal", "Negotiation", "Closed-Won", "Closed-Lost"]
    HEALTH = ["Positive", "Neutral", "Negative"]
    GROWTH = ["Startup", "Scaleup", "Enterprise"]

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.

        If `seed` is provided both Python's `random` and `Faker` are seeded so
        repeated runs are deterministic.
        """
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)

        # Trackers represent topical keywords per industry used in emails/meetings
        self.trackers_by_industry = {
            "SaaS": ["SSO", "API", "SOC2", "uptime", "rate-limits"],
            "Healthcare": ["HIPAA", "PHI", "EMR", "integration", "VPN"],
            "Financial Services": ["KYC", "AML", "PCI", "compliance", "SLA"],
            "Manufacturing": ["OT", "PLC", "supply-chain", "ERP", "IoT"]
        }

    def gen_companies(self, n: int, industries: List[str]) -> List[Company]:
        """Generate `n` companies distributed among the given `industries`.

        Company size and revenue are sampled to create realistic spreads and to
        later scale deal values.
        """
        companies = []
        for _ in range(n):
            industry = random.choice(industries)
            # coarse employee count buckets: small, mid, large
            size = random.choices([50, 200, 2000], weights=[0.4, 0.4, 0.2])[0]
            growth = "Startup" if size < 100 else ("Scaleup" if size < 1000 else "Enterprise")
            # revenue roughly proportional to size with some randomness
            revenue = size * random.randint(80000, 200000)
            c = Company(
                company_id=IDS(),
                name=fake.company(),
                industry=industry,
                employee_count=size,
                growth_stage=growth,
                hq_region=fake.country(),
                annual_revenue_usd=revenue,
                context_note=fake.sentence(nb_words=10)
            )
            companies.append(c)
        return companies

    def gen_contacts(self, companies: List[Company], per_company=3) -> List[Contact]:
        """Generate contacts for each company.

        `per_company` controls how many contacts to create per company. Titles and
        seniority are sampled to create a realistic mix.
        """
        contacts = []
        titles = ["Engineer", "Manager", "Director", "VP", "CIO"]
        seniorities = ["IC", "Manager", "Director", "VP", "C-Level"]
        for c in companies:
            for i in range(per_company):
                name = fake.name()
                senior = random.choices(seniorities, weights=[30,25,20,15,10])[0]
                cont = Contact(
                    contact_id=IDS(),
                    company_id=c.company_id,
                    full_name=name,
                    title=random.choice(titles),
                    seniority=senior,
                    email=fake.company_email(),
                    relationship_note=f"Previously worked with {fake.company()} on {fake.word()}"
                )
                contacts.append(cont)
        return contacts

    def gen_reps(self, n=5) -> List[SalesRep]:
        """Generate a small pool of sales reps with tiered targets."""
        tiers = ["Top", "Good", "Average", "Underperformer"]
        targets = {"Top": random.randint(8,12), "Good": random.randint(5,8), "Average": random.randint(3,5), "Underperformer": random.randint(1,3)}
        reps = []
        for _ in range(n):
            tier = random.choice(tiers)
            reps.append(SalesRep(rep_id=IDS(), full_name=fake.name(), tier=tier, quarter_deals_closed_target=targets[tier]))
        return reps

    def gen_deals(self, companies: List[Company], contacts: List[Contact], reps: List[SalesRep]) -> List[Deal]:
        """Generate deals for the provided companies.

        Each company may have 0-2 deals; deal `value_usd` is scaled roughly by
        company size to keep outputs internally consistent.
        """
        deals = []
        now = datetime.utcnow()
        for c in companies:
            # Each company may have 0-2 deals
            for _ in range(random.randint(0,2)):
                # pick a contact that belongs to this company
                contact = random.choice([ct for ct in contacts if ct.company_id == c.company_id])
                rep = random.choice(reps)
                stage = random.choice(self.STAGES)
                health = random.choices(self.HEALTH, weights=[40,40,20])[0]
                # opened_at is some time in the recent past
                opened = now - timedelta(days=random.randint(0,60))
                expected_close = opened + timedelta(days=random.randint(30,90)) if stage not in ["Closed-Won","Closed-Lost"] else None
                closed_at = None
                loss_reason = None
                if stage == "Closed-Won":
                    closed_at = (opened + timedelta(days=random.randint(10,90))).isoformat()
                if stage == "Closed-Lost":
                    closed_at = (opened + timedelta(days=random.randint(1,90))).isoformat()
                    loss_reason = random.choice(["Budget", "Timing", "Competitive"])
                # value scaled by employee_count buckets
                multiplier = 1000 if c.employee_count < 500 else (10000 if c.employee_count < 1500 else 50000)
                value = multiplier * random.randint(1,10)
                d = Deal(
                    deal_id=IDS(),
                    company_id=c.company_id,
                    primary_contact_id=contact.contact_id,
                    rep_id=rep.rep_id,
                    stage=stage,
                    health=health,
                    value_usd=value,
                    opened_at=opened.isoformat(),
                    expected_close_at=expected_close.isoformat() if expected_close else None,
                    closed_at=closed_at,
                    loss_reason=loss_reason
                )
                deals.append(d)
        return deals

    def gen_emails(self, deals: List[Deal], contacts: List[Contact], reps: List[SalesRep]) -> List[Email]:
        """Generate email threads for each deal.

        The number of messages and reply latencies vary by deal stage and health
        to simulate realistic cadences (faster replies for positive deals).
        """
        emails = []
        for d in deals:
            # decide number of messages in a single thread by stage
            if d.stage in ["Prospecting"]:
                msgs = random.randint(3,5)
            elif d.stage in ["Qualified","Demo"]:
                msgs = random.randint(5,10)
            else:
                msgs = random.randint(8,20)
            thread_id = IDS()
            # start the thread at the deal's opened_at time
            last_ts = datetime.fromisoformat(d.opened_at)
            prev_email_id = None
            for idx in range(1, msgs+1):
                # alternate direction with more outbound initially
                if idx % 3 == 0:
                    direction = "inbound"
                else:
                    direction = "outbound"
                sender = d.primary_contact_id if direction == "inbound" else d.rep_id
                recipients = [d.rep_id] if direction == "inbound" else [d.primary_contact_id]
                # inbound reply latency modeled by deal health
                if d.health == "Positive":
                    latency = random.randint(1,48) if direction == "inbound" else None
                elif d.health == "Neutral":
                    latency = random.randint(48,72) if direction == "inbound" else None
                else:
                    latency = random.randint(120,240) if direction == "inbound" else None
                if latency:
                    last_ts = last_ts + timedelta(hours=latency)
                else:
                    last_ts = last_ts + timedelta(hours=random.randint(6,48))
                subject = f"{d.stage} discussion - {fake.bs().title()}"
                body = fake.paragraph(nb_sentences=3)
                sentiment = random.choices(["Positive","Neutral","Negative"], weights=self._sentiment_weights(d.health))[0]
                trackers = self._pick_trackers_for_deal(d)
                email = Email(
                    email_id=IDS(),
                    deal_id=d.deal_id,
                    thread_id=thread_id,
                    direction=direction,
                    sender_id=sender,
                    recipient_ids=recipients,
                    timestamp=last_ts.isoformat(),
                    subject=subject,
                    body=body,
                    sequence_index=idx,
                    sentiment=sentiment,
                    trackers=trackers,
                    reply_latency_hours=latency,
                    in_reply_to_email_id=prev_email_id,
                    outcome=random.choice(["replied","ignored","forwarded","objection","scheduling-request"])
                )
                prev_email_id = email.email_id
                emails.append(email)
        return emails

    def gen_meetings(self, deals: List[Deal], contacts: List[Contact], reps: List[SalesRep]) -> List[Meeting]:
        """Generate meetings tied to deals, modeling a typical sales cadence.

        Meetings are created from a short canonical path (Discovery -> Demo -> ...)
        and scheduled relative to the deal opened date.
        """
        meetings = []
        for d in deals:
            # meetings follow a canonical path
            path = ["Discovery Call", "Demo", "Technical Deep Dive", "Proposal Review", "Final Decision"]
            n = random.randint(0, min(4, len(path)-1))
            base = datetime.fromisoformat(d.opened_at)
            for i in range(n):
                start = base + timedelta(days=5*(i+1) + random.randint(0,5))
                end = start + timedelta(hours=1)
                attendees = [d.rep_id, d.primary_contact_id]
                title = path[i]
                sentiment = random.choices(["Positive","Neutral","Negative"], weights=self._meeting_weights(d.health))[0]
                notes = f"Discussed {title}. References email thread and next steps. {fake.sentence()}"
                meeting = Meeting(
                    meeting_id=IDS(),
                    deal_id=d.deal_id,
                    title=title,
                    stage_at_time=d.stage,
                    scheduled_start=start.isoformat(),
                    scheduled_end=end.isoformat(),
                    actual_start=start.isoformat(),
                    actual_end=end.isoformat(),
                    attendee_ids=attendees,
                    outcome=random.choice(["completed","no-show","rescheduled","decision-meeting","blocked"]),
                    notes=notes,
                    sentiment=sentiment,
                    trackers=self._pick_trackers_for_deal(d),
                    follow_up_action=random.choice(["Send SOC2 report","Share API rate limits","Draft MSA","Provide integration docs"]) 
                )
                meetings.append(meeting)
        return meetings

    def _pick_trackers_for_deal(self, deal: Deal) -> List[str]:
        # naive: choose trackers by company industry if possible
        # this function expects that we can map deal->company via external context; if not, return a mixed set
        choices = sum(self.trackers_by_industry.values(), [])
        return random.sample(choices, k=min(3, len(choices)))

    def _sentiment_weights(self, health: str):
        if health == "Positive":
            return [60,30,10]
        if health == "Neutral":
            return [30,40,30]
        return [15,35,50]

    def _meeting_weights(self, health: str):
        if health == "Positive":
            return [60,35,5]
        if health == "Neutral":
            return [30,50,20]
        return [10,40,50]

    def to_json(self, objs: List) -> List[dict]:
        return [asdict(o) for o in objs]
