"""CLI entry point for generating ProspectIQ demo data"""
import argparse
import json
import os
from typing import List

from .core import Generator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--companies", type=int, default=10)
    p.add_argument("--industries", type=str, default="SaaS,Healthcare,Financial Services,Manufacturing")
    p.add_argument("--outdir", type=str, default="output")
    return p.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    industries = [s.strip() for s in args.industries.split(",") if s.strip()]
    g = Generator(seed=args.seed)
    companies = g.gen_companies(args.companies, industries)
    contacts = g.gen_contacts(companies, per_company=3)
    reps = g.gen_reps(n=6)
    deals = g.gen_deals(companies, contacts, reps)
    emails = g.gen_emails(deals, contacts, reps)
    meetings = g.gen_meetings(deals, contacts, reps)

    ensure_dir(args.outdir)
    write_json(os.path.join(args.outdir, "companies.json"), g.to_json(companies))
    write_json(os.path.join(args.outdir, "contacts.json"), g.to_json(contacts))
    write_json(os.path.join(args.outdir, "sales_reps.json"), g.to_json(reps))
    write_json(os.path.join(args.outdir, "deals.json"), g.to_json(deals))
    write_json(os.path.join(args.outdir, "emails.json"), g.to_json(emails))
    write_json(os.path.join(args.outdir, "meetings.json"), g.to_json(meetings))

    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
