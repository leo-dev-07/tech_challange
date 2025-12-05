"""Basic validation checks for generated outputs"""
import argparse
import json
import os
from collections import defaultdict


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", type=str, default="output")
    return p.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    files = {n: load_json(os.path.join(indir, f)) for n, f in [
        ("companies","companies.json"), ("contacts","contacts.json"), ("reps","sales_reps.json"), ("deals","deals.json"), ("emails","emails.json"), ("meetings","meetings.json")
    ]}

    # Relational integrity
    ids = defaultdict(set)
    for c in files["companies"]:
        ids["company"].add(c["company_id"])
    for ct in files["contacts"]:
        ids["contact"].add(ct["contact_id"])
    for r in files["reps"]:
        ids["rep"].add(r["rep_id"])

    ok = True
    for d in files["deals"]:
        if d["company_id"] not in ids["company"]:
            print("Broken FK: deal.company_id", d)
            ok = False
        if d["primary_contact_id"] not in ids["contact"]:
            print("Broken FK: deal.primary_contact_id", d)
            ok = False
        if d["rep_id"] not in ids["rep"]:
            print("Broken FK: deal.rep_id", d)
            ok = False

    # thread logic: sequence_index increasing per thread
    threads = defaultdict(list)
    for e in files["emails"]:
        threads[e["thread_id"]].append(e)
    for tid, msgs in threads.items():
        msgs_sorted = sorted(msgs, key=lambda x: x["sequence_index"])
        for i, m in enumerate(msgs_sorted, start=1):
            if m["sequence_index"] != i:
                print("Thread sequence gap", tid)
                ok = False

    if ok:
        print("Basic checks passed")
    else:
        print("Some checks failed")


if __name__ == "__main__":
    main()
