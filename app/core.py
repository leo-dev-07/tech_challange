from __future__ import annotations
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4
from faker import Faker

fake = Faker()

IDS = lambda: str(uuid4())

@dataclass
class Company:
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
    contact_id: str
    company_id: str
    full_name: str
    title: str
    seniority: str
    email: str
    relationship_note: str

@dataclass
class SalesRep:
    rep_id: str
    full_name: str
    tier: str
    quarter_deals_closed_target: int

@dataclass
class Deal:
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
    STAGES = ["Prospecting", "Qualified", "Demo", "Proposal", "Negotiation", "Closed-Won", "Closed-Lost"]
    HEALTH = ["Positive", "Neutral", "Negative"]
    GROWTH = ["Startup", "Scaleup", "Enterprise"]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        self.trackers_by_industry = {
            "SaaS": ["SSO", "API", "SOC2", "uptime", "rate-limits"],
            "Healthcare": ["HIPAA", "PHI", "EMR", "integration", "VPN"],
            "Financial Services": ["KYC", "AML", "PCI", "compliance", "SLA"],
            "Manufacturing": ["OT", "PLC", "supply-chain", "ERP", "IoT"]
        }

    def gen_companies(self, n: int, industries: List[str]) -> List[Company]:
        companies = []
        for _ in range(n):
            industry = random.choice(industries)
            size = random.choices([50, 200, 2000], weights=[0.4, 0.4, 0.2])[0]
            growth = "Startup" if size < 100 else ("Scaleup" if size < 1000 else "Enterprise")
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
        tiers = ["Top", "Good", "Average", "Underperformer"]
        targets = {"Top": random.randint(8,12), "Good": random.randint(5,8), "Average": random.randint(3,5), "Underperformer": random.randint(1,3)}
        reps = []
        for _ in range(n):
            tier = random.choice(tiers)
            reps.append(SalesRep(rep_id=IDS(), full_name=fake.name(), tier=tier, quarter_deals_closed_target=targets[tier]))
        return reps

    def gen_deals(self, companies: List[Company], contacts: List[Contact], reps: List[SalesRep]) -> List[Deal]:
        deals = []
        now = datetime.utcnow()
        for c in companies:
            # Each company may have 0-2 deals
            for _ in range(random.randint(0,2)):
                contact = random.choice([ct for ct in contacts if ct.company_id == c.company_id])
                rep = random.choice(reps)
                stage = random.choice(self.STAGES)
                health = random.choices(self.HEALTH, weights=[40,40,20])[0]
                opened = now - timedelta(days=random.randint(0,60))
                expected_close = opened + timedelta(days=random.randint(30,90)) if stage not in ["Closed-Won","Closed-Lost"] else None
                closed_at = None
                loss_reason = None
                if stage == "Closed-Won":
                    closed_at = (opened + timedelta(days=random.randint(10,90))).isoformat()
                if stage == "Closed-Lost":
                    closed_at = (opened + timedelta(days=random.randint(1,90))).isoformat()
                    loss_reason = random.choice(["Budget", "Timing", "Competitive"])
                # value scaled by employee_count
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
        emails = []
        for d in deals:
            # decide number of threads and messages per thread by stage
            if d.stage in ["Prospecting"]:
                msgs = random.randint(3,5)
            elif d.stage in ["Qualified","Demo"]:
                msgs = random.randint(5,10)
            else:
                msgs = random.randint(8,20)
            thread_id = IDS()
            last_ts = datetime.fromisoformat(d.opened_at)
            prev_email_id = None
            for idx in range(1, msgs+1):
                # alternate direction with higher outbound early
                if idx % 3 == 0:
                    direction = "inbound"
                else:
                    direction = "outbound"
                sender = d.primary_contact_id if direction == "inbound" else d.rep_id
                recipients = [d.rep_id] if direction == "inbound" else [d.primary_contact_id]
                # latency depends on health
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
        meetings = []
        for d in deals:
            # meetings follow canonical path maybe
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
