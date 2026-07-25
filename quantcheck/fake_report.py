"""
fake_report.py

Generates a synthetic "detailed report" containing precise, checkable facts
(numbers, dates, names, IDs) embedded in natural-language prose. Also emits
a ground_truth.json mapping specific questions to the exact expected answers
and which chunk they live in.

This is the "ground truth" that embed_and_index.py will chunk/embed, and that
query_test.py will use to check whether TurboVec-compressed retrieval still
surfaces the right chunk for a precise fact question (vs. the full-precision
baseline index).

Deterministic (fixed seed) so results are reproducible across bit_width runs.

Output:
  report.txt        - the full synthetic report, chunk-delimited
  ground_truth.json - list of {id, question, answer, chunk_id, fact_type}
"""

import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SEED = 42
random.seed(SEED)

FIRST_NAMES = [
    "Marek", "Kaisa", "Priit", "Liisa", "Andres", "Kadri", "Toomas", "Mari",
    "Rein", "Piret", "Jaan", "Anu", "Meelis", "Kersti", "Indrek", "Terje",
]
LAST_NAMES = [
    "Tamm", "Saar", "Kask", "Sepp", "Mägi", "Kukk", "Vaher", "Org",
    "Laine", "Pärn", "Kalda", "Rebane",
]
SERVER_LOCATIONS = ["Tallinn-DC1", "Tartu-DC2", "Helsinki-DC3", "Frankfurt-DC4"]
INCIDENT_TYPES = [
    "disk saturation", "memory leak", "network partition",
    "certificate expiry", "OOM kill", "cold cache stampede",
]

CHUNKS = []
FACTS = []
fact_counter = 0


def add_fact(question, answer, chunk_id, fact_type):
    global fact_counter
    fact_counter += 1
    FACTS.append({
        "id": f"F{fact_counter:03d}",
        "question": question,
        "answer": str(answer),
        "chunk_id": chunk_id,
        "fact_type": fact_type,
    })


def rand_person():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def rand_date():
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"2026-{month:02d}-{day:02d}"


# ---------------------------------------------------------------------------
# Section builders. Each returns prose text for one chunk and registers the
# exact facts it contains via add_fact().
# ---------------------------------------------------------------------------

def build_incident_chunk(chunk_id, incident_num):
    person = rand_person()
    itype = random.choice(INCIDENT_TYPES)
    location = random.choice(SERVER_LOCATIONS)
    date = rand_date()
    duration_min = random.randint(4, 240)
    severity = random.choice([1, 2, 3, 4, 5])
    ticket_id = f"INC-{random.randint(10000, 99999)}"

    text = (
        f"Incident Report #{incident_num} (Ticket {ticket_id})\n"
        f"On {date}, server cluster {location} experienced a {itype} event. "
        f"The on-call engineer, {person}, was paged and resolved the incident "
        f"after {duration_min} minutes. Severity was classified as level {severity} "
        f"out of 5. Root cause analysis was filed the following business day."
    )

    add_fact(f"What was the ticket ID for Incident Report #{incident_num}?", ticket_id, chunk_id, "id")
    add_fact(f"Who was the on-call engineer for Incident Report #{incident_num}?", person, chunk_id, "name")
    add_fact(f"What date did Incident Report #{incident_num} occur?", date, chunk_id, "date")
    add_fact(f"How many minutes did it take to resolve Incident Report #{incident_num}?", duration_min, chunk_id, "number")
    add_fact(f"What severity level was Incident Report #{incident_num}?", severity, chunk_id, "number")
    add_fact(f"Which data center was affected in Incident Report #{incident_num}?", location, chunk_id, "name")

    return text


def build_financial_chunk(chunk_id, quarter_label):
    revenue = round(random.uniform(80_000, 950_000), 2)
    opex = round(revenue * random.uniform(0.35, 0.7), 2)
    headcount = random.randint(12, 140)
    churn_pct = round(random.uniform(0.5, 9.9), 1)
    approver = rand_person()

    text = (
        f"Financial Summary — {quarter_label}\n"
        f"Total revenue for {quarter_label} was EUR {revenue:,.2f}. Operating "
        f"expenses came in at EUR {opex:,.2f}. Headcount at quarter close was "
        f"{headcount} employees. Customer churn was measured at {churn_pct}%. "
        f"The budget was approved by {approver}."
    )

    add_fact(f"What was total revenue for {quarter_label}?", f"EUR {revenue:,.2f}", chunk_id, "number")
    add_fact(f"What were operating expenses for {quarter_label}?", f"EUR {opex:,.2f}", chunk_id, "number")
    add_fact(f"What was headcount at the close of {quarter_label}?", headcount, chunk_id, "number")
    add_fact(f"What was the customer churn percentage for {quarter_label}?", f"{churn_pct}%", chunk_id, "number")
    add_fact(f"Who approved the budget for {quarter_label}?", approver, chunk_id, "name")

    return text


def build_personnel_chunk(chunk_id):
    person = rand_person()
    role = random.choice([
        "Site Reliability Engineer", "Backend Developer", "Data Platform Lead",
        "Security Analyst", "Infrastructure Architect", "QA Engineer",
    ])
    start_date = rand_date()
    employee_id = f"EMP-{random.randint(1000, 9999)}"
    manager = rand_person()

    text = (
        f"Personnel Record ({employee_id})\n"
        f"{person} joined as {role} on {start_date}. Their reporting manager "
        f"is {manager}. Employee ID {employee_id} was issued on the same date "
        f"as their start date."
    )

    add_fact(f"What is the employee ID for {person}?", employee_id, chunk_id, "id")
    add_fact(f"What role did {person} join as?", role, chunk_id, "name")
    add_fact(f"When did {person} start ({employee_id})?", start_date, chunk_id, "date")
    add_fact(f"Who is {person}'s reporting manager?", manager, chunk_id, "name")

    return text


def build_server_spec_chunk(chunk_id):
    location = random.choice(SERVER_LOCATIONS)
    rack_id = f"RACK-{random.randint(1, 40):02d}"
    cpu_cores = random.choice([32, 48, 64, 96, 128])
    ram_gb = random.choice([128, 256, 512, 768])
    uptime_days = random.randint(10, 900)

    text = (
        f"Server Specification for {rack_id} ({location})\n"
        f"Rack {rack_id} at {location} runs {cpu_cores} CPU cores and "
        f"{ram_gb} GB of RAM. Current uptime is {uptime_days} days without "
        f"a reboot."
    )

    add_fact(f"How many CPU cores does {rack_id} have?", cpu_cores, chunk_id, "number")
    add_fact(f"How much RAM does {rack_id} have?", f"{ram_gb} GB", chunk_id, "number")
    add_fact(f"What is the current uptime of {rack_id}?", f"{uptime_days} days", chunk_id, "number")
    add_fact(f"Which location hosts {rack_id}?", location, chunk_id, "name")

    return text


# ---------------------------------------------------------------------------
# Assemble the report
# ---------------------------------------------------------------------------

def build_report(num_incidents=6, num_financial=4, num_personnel=6, num_servers=6):
    chunk_id = 0
    lines = [
        "SYNTHETIC INFRASTRUCTURE & OPERATIONS REPORT",
        "Generated for quantcheck retrieval-fidelity testing.",
        "=" * 60,
        "",
    ]

    for i in range(1, num_incidents + 1):
        chunk_id += 1
        cid = f"C{chunk_id:03d}"
        lines.append(f"[[chunk:{cid}]]")
        lines.append(build_incident_chunk(cid, i))
        lines.append("")

    quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"][:num_financial]
    for q in quarters:
        chunk_id += 1
        cid = f"C{chunk_id:03d}"
        lines.append(f"[[chunk:{cid}]]")
        lines.append(build_financial_chunk(cid, q))
        lines.append("")

    for _ in range(num_personnel):
        chunk_id += 1
        cid = f"C{chunk_id:03d}"
        lines.append(f"[[chunk:{cid}]]")
        lines.append(build_personnel_chunk(cid))
        lines.append("")

    for _ in range(num_servers):
        chunk_id += 1
        cid = f"C{chunk_id:03d}"
        lines.append(f"[[chunk:{cid}]]")
        lines.append(build_server_spec_chunk(cid))
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report_text = build_report()

    report_path = os.path.join(OUT_DIR, "report.txt")
    gt_path = os.path.join(OUT_DIR, "ground_truth.json")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "facts": FACTS}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {report_path} ({len(report_text.splitlines())} lines, "
          f"{report_text.count('[[chunk:')} chunks)")
    print(f"Wrote {gt_path} ({len(FACTS)} fact questions)")


if __name__ == "__main__":
    main()
