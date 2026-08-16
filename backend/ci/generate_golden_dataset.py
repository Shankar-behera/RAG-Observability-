"""
Generates backend/ci/corpus.json (40 ACL-tagged documents across
Engineering/HR/Executive/General) and backend/ci/golden_dataset.json
(50 query/role/ground_truth triples used by the CI regression gate).

Run once to (re)materialize the fixtures:
    python backend/ci/generate_golden_dataset.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

CORPUS = [
    # ---- Engineering (10) ----
    ("Engineering", "deploys.md", "Deployments to production require a passing CI pipeline, one approving code review, and a rollback plan documented in the PR description. Deploys happen Tuesday-Thursday, 10am-4pm, to avoid weekend on-call load."),
    ("Engineering", "oncall.md", "On-call rotates weekly across the platform team. The on-call engineer carries the pager, triages incoming alerts within 15 minutes, and escalates to the incident commander if a Sev1 is not mitigated within 30 minutes."),
    ("Engineering", "code_review.md", "Every pull request needs at least one approval before merge. PRs touching authentication, billing, or the data pipeline require a second approval from a senior engineer on that subsystem."),
    ("Engineering", "incident_response.md", "Sev1 incidents trigger an automatic incident channel, page the on-call engineer and the incident commander, and require a postmortem within 5 business days of resolution."),
    ("Engineering", "cicd.md", "The CI/CD pipeline runs unit tests, integration tests, and a security scan on every push. Merges to main auto-deploy to staging; production deploys require a manual approval gate."),
    ("Engineering", "db_migrations.md", "Database migrations must be backward-compatible and reviewed by the data platform team. Destructive migrations (column drops, table drops) require a two-week deprecation window."),
    ("Engineering", "api_versioning.md", "Public APIs are versioned via the URL path (/v1, /v2). A version is supported for at least 12 months after the next version ships, with deprecation notices sent 90 days before sunset."),
    ("Engineering", "tech_stack.md", "The core platform runs on Python/FastAPI services, a PostgreSQL primary datastore, Redis for caching, and Kafka for event streaming. The frontend is a Next.js/TypeScript monorepo."),
    ("Engineering", "security_patching.md", "Critical CVEs affecting production dependencies must be patched within 72 hours of disclosure. High-severity CVEs get a 2-week SLA, and are tracked in the security backlog."),
    ("Engineering", "testing_strategy.md", "New features require unit test coverage above 80% for the touched module and at least one integration test exercising the primary user flow before merge."),

    # ---- HR (10) ----
    ("HR", "payroll.md", "Payroll runs on the last business day of each month via direct deposit. Pay stubs are available in the HR portal within 24 hours of the pay run."),
    ("HR", "leave_policy.md", "Full-time employees accrue 18 days of paid time off per year, accrued monthly. Unused PTO up to 5 days carries over into the next calendar year."),
    ("HR", "benefits.md", "Open enrollment for health, dental, and vision benefits runs each November for coverage starting January 1. New hires have 30 days from their start date to enroll."),
    ("HR", "remote_work.md", "Employees may work remotely up to 3 days per week with manager approval. Fully remote arrangements require VP-level sign-off and are reviewed annually."),
    ("HR", "onboarding.md", "New hires complete a 2-week onboarding program covering security training, benefits enrollment, and team-specific ramp-up, tracked via the onboarding checklist in the HR portal."),
    ("HR", "performance_review.md", "Performance reviews happen twice yearly, in June and December. Each review includes self-assessment, manager assessment, and a calibration session across the team."),
    ("HR", "expense_reimbursement.md", "Business expenses are reimbursed within 10 business days of submission with a receipt. Expenses over $500 require manager pre-approval."),
    ("HR", "parental_leave.md", "Primary caregivers receive 16 weeks of paid parental leave; secondary caregivers receive 8 weeks. Leave can be taken continuously or split within the first year."),
    ("HR", "harassment_policy.md", "The company maintains a zero-tolerance policy for harassment and discrimination. Reports can be filed confidentially through the HR portal or the anonymous ethics hotline."),
    ("HR", "referral_program.md", "Employees who refer a successful hire receive a $3,000 bonus, paid after the new hire completes 90 days, split 50/50 between referral submission and the 90-day milestone."),

    # ---- Executive (10) ----
    ("Executive", "q3_strategy.md", "Q3 strategic priorities are expanding the mid-market sales motion, shipping the v2 platform architecture, and closing the Series C financing round by end of quarter."),
    ("Executive", "board_minutes.md", "The board approved the FY25 budget with a 22% increase in R&D headcount and flagged customer concentration risk as a standing agenda item for quarterly review."),
    ("Executive", "ma_pipeline.md", "The M&A pipeline currently includes two early-stage acquisition targets in the observability tooling space, both under NDA and pending due diligence."),
    ("Executive", "exec_compensation.md", "Executive compensation bands are set annually by the compensation committee, benchmarked against a peer group of similarly-staged companies, and include a base, bonus, and equity component."),
    ("Executive", "budget_allocation.md", "FY25 budget allocates 45% to R&D, 25% to sales and marketing, 15% to G&A, and 15% held as strategic reserve for opportunistic hiring or acquisitions."),
    ("Executive", "market_expansion.md", "The market expansion plan targets EMEA as the first international market, with a local sales hire and data residency compliance work starting in Q4."),
    ("Executive", "competitive_analysis.md", "The primary competitive threat is a well-funded incumbent with broader enterprise distribution; our differentiation is faster time-to-value and usage-based pricing."),
    ("Executive", "risk_management.md", "The enterprise risk register tracks customer concentration, key-person dependency in the platform team, and vendor lock-in with the primary cloud provider as top risks."),
    ("Executive", "investor_relations.md", "Investor updates go out monthly to the board and quarterly to all shareholders, covering ARR growth, burn rate, and runway."),
    ("Executive", "succession_planning.md", "Succession plans are on file for all C-suite roles, reviewed annually with the board, identifying at least one internal and one external potential successor per role."),

    # ---- General (10) ----
    ("General", "office_hours.md", "The office is open 7am-8pm weekdays with badge access. Core collaboration hours are 10am-4pm local time, when most teams expect availability for meetings."),
    ("General", "holiday_calendar.md", "The company observes 11 paid holidays annually, published each January in the HR portal, plus two floating holidays employees can schedule at their discretion."),
    ("General", "it_helpdesk.md", "IT support tickets can be filed through the helpdesk portal or by emailing it-support@company.com. Standard response SLA is 4 business hours."),
    ("General", "parking.md", "Employee parking is available in the garage on a first-come basis on levels 2-4. EV charging spots are on level 2 and require registration through facilities."),
    ("General", "cafeteria.md", "The cafeteria serves breakfast 7-10am and lunch 11:30am-2pm, with a rotating weekly menu posted every Monday on the office intranet."),
    ("General", "mission_statement.md", "The company mission is to make production software observable and trustworthy by default, so teams can ship faster without sacrificing reliability."),
    ("General", "brand_guidelines.md", "Brand guidelines specify the primary logo, color palette, and typography for all external-facing materials; the design team reviews external decks before publication."),
    ("General", "room_booking.md", "Conference rooms are booked through the shared calendar system. Meetings longer than 2 hours require facilities approval to hold a room past standard slots."),
    ("General", "visitor_policy.md", "All visitors must be signed in at reception, badged, and escorted by their host at all times while in the building."),
    ("General", "evacuation.md", "In case of a fire alarm, use the nearest stairwell (elevators are disabled), proceed to the assembly point across the street, and check in with your floor warden."),
]

# Golden dataset: 50 (query, role, ground_truth) triples.
# `expected_source` documents which corpus doc should ground the answer, used
# by CI to sanity-check retrieval even before Ragas scoring.
GOLDEN: list[dict] = []


def add(query, role, ground_truth, expected_source):
    GOLDEN.append({"query": query, "role": role, "ground_truth": ground_truth, "expected_source": expected_source})


# Engineering role queries (12)
add("What approvals are needed before a production deploy?", "Engineering", "A passing CI pipeline and one approving code review, plus a documented rollback plan.", "deploys.md")
add("How often does on-call rotate?", "Engineering", "On-call rotates weekly across the platform team.", "oncall.md")
add("Do billing-related PRs need extra review?", "Engineering", "Yes, PRs touching authentication, billing, or the data pipeline require a second approval from a senior engineer.", "code_review.md")
add("When is a postmortem required after an incident?", "Engineering", "Within 5 business days of resolving a Sev1 incident.", "incident_response.md")
add("What does the CI/CD pipeline run on every push?", "Engineering", "Unit tests, integration tests, and a security scan.", "cicd.md")
add("What's the deprecation window for a destructive database migration?", "Engineering", "A two-week deprecation window is required for destructive migrations.", "db_migrations.md")
add("How long is an API version supported after a new version ships?", "Engineering", "At least 12 months, with deprecation notices sent 90 days before sunset.", "api_versioning.md")
add("What database does the platform use?", "Engineering", "PostgreSQL is the primary datastore.", "tech_stack.md")
add("What's the SLA for patching a critical CVE?", "Engineering", "Critical CVEs must be patched within 72 hours of disclosure.", "security_patching.md")
add("What's the minimum unit test coverage for a new feature?", "Engineering", "Coverage above 80% for the touched module.", "testing_strategy.md")
add("Can I read the office holiday calendar as an engineer?", "Engineering", "Yes, the company observes 11 paid holidays annually plus two floating holidays.", "holiday_calendar.md")
add("What are core collaboration hours in the office?", "Engineering", "10am-4pm local time.", "office_hours.md")

# HR role queries (12)
add("When does payroll run each month?", "HR", "Payroll runs on the last business day of each month via direct deposit.", "payroll.md")
add("How much PTO do full-time employees accrue per year?", "HR", "18 days per year, accrued monthly, with up to 5 days carrying over.", "leave_policy.md")
add("When is open enrollment for benefits?", "HR", "Each November, for coverage starting January 1.", "benefits.md")
add("How many remote days are employees allowed per week?", "HR", "Up to 3 days per week with manager approval.", "remote_work.md")
add("How long is the onboarding program?", "HR", "A 2-week onboarding program.", "onboarding.md")
add("How often are performance reviews conducted?", "HR", "Twice yearly, in June and December.", "performance_review.md")
add("How long does expense reimbursement take?", "HR", "Within 10 business days of submission with a receipt.", "expense_reimbursement.md")
add("How many weeks of parental leave do primary caregivers get?", "HR", "16 weeks of paid parental leave for primary caregivers.", "parental_leave.md")
add("How can harassment be reported confidentially?", "HR", "Through the HR portal or the anonymous ethics hotline.", "harassment_policy.md")
add("What is the employee referral bonus amount?", "HR", "$3,000, split 50/50 between referral submission and the 90-day milestone.", "referral_program.md")
add("Can I check the IT helpdesk contact as an HR employee?", "HR", "Yes, tickets can be filed via the helpdesk portal or it-support@company.com.", "it_helpdesk.md")
add("What are the cafeteria hours?", "HR", "Breakfast 7-10am and lunch 11:30am-2pm.", "cafeteria.md")

# Executive role queries (13) - includes cross-cutting access to Eng/HR/General too
add("What are the Q3 strategic priorities?", "Executive", "Expanding mid-market sales, shipping the v2 platform architecture, and closing the Series C round.", "q3_strategy.md")
add("What did the board approve for the FY25 budget?", "Executive", "A 22% increase in R&D headcount, with customer concentration flagged as a standing risk.", "board_minutes.md")
add("What's in the M&A pipeline right now?", "Executive", "Two early-stage acquisition targets in the observability tooling space, under NDA.", "ma_pipeline.md")
add("Who sets executive compensation bands?", "Executive", "The compensation committee, benchmarked against a peer group, annually.", "exec_compensation.md")
add("How is the FY25 budget allocated across departments?", "Executive", "45% R&D, 25% sales and marketing, 15% G&A, 15% strategic reserve.", "budget_allocation.md")
add("What market is targeted first for international expansion?", "Executive", "EMEA, starting with a local sales hire and data residency compliance work in Q4.", "market_expansion.md")
add("What is our main competitive differentiation?", "Executive", "Faster time-to-value and usage-based pricing versus a well-funded incumbent.", "competitive_analysis.md")
add("What are the top risks on the enterprise risk register?", "Executive", "Customer concentration, key-person dependency, and cloud vendor lock-in.", "risk_management.md")
add("How often do investor updates go out?", "Executive", "Monthly to the board, quarterly to all shareholders.", "investor_relations.md")
add("Is there a succession plan for the CEO role?", "Executive", "Yes, succession plans are on file for all C-suite roles, reviewed annually with the board.", "succession_planning.md")
add("As an executive, what approvals are needed for a production deploy?", "Executive", "A passing CI pipeline and one approving code review, plus a documented rollback plan.", "deploys.md")
add("As an executive, how much PTO do employees accrue?", "Executive", "18 days per year, accrued monthly.", "leave_policy.md")
add("As an executive, what's the visitor sign-in policy?", "Executive", "All visitors are signed in at reception, badged, and escorted by their host at all times.", "visitor_policy.md")

# General/shared queries usable by any role (13)
add("Where can I park at the office?", "General", "In the garage on levels 2-4, first-come basis, with EV charging on level 2.", "parking.md")
add("What should I do during a fire alarm?", "General", "Use the nearest stairwell, proceed to the assembly point, and check in with your floor warden.", "evacuation.md")
add("What is the company mission statement?", "General", "To make production software observable and trustworthy by default.", "mission_statement.md")
add("How do I book a conference room?", "General", "Through the shared calendar system; meetings over 2 hours need facilities approval.", "room_booking.md")
add("What are the brand guidelines used for?", "General", "They specify logo, color palette, and typography for external-facing materials.", "brand_guidelines.md")
add("What's the visitor policy at the office?", "General", "Visitors sign in at reception, get badged, and are escorted by their host at all times.", "visitor_policy.md")
add("What are the office hours?", "General", "7am-8pm weekdays with badge access.", "office_hours.md")
add("How many paid holidays does the company observe?", "General", "11 paid holidays annually plus two floating holidays.", "holiday_calendar.md")
add("How do I contact IT support?", "General", "File a ticket through the helpdesk portal or email it-support@company.com.", "it_helpdesk.md")
add("What time does the cafeteria serve lunch?", "General", "11:30am-2pm.", "cafeteria.md")
add("As a General-access employee, can I see the Q3 executive strategy?", "General", "I don't have enough information to answer that.", None)
add("As a General-access employee, can I see the payroll schedule?", "General", "I don't have enough information to answer that.", None)
add("As a General-access employee, can I see engineering's on-call rotation details?", "General", "I don't have enough information to answer that.", None)

assert len(GOLDEN) == 50, f"expected exactly 50 golden pairs, got {len(GOLDEN)}"
assert len(CORPUS) == 40, f"expected exactly 40 corpus docs, got {len(CORPUS)}"

corpus_json = [
    {"acl_role": role, "source": source, "text": text}
    for role, source, text in CORPUS
]

(OUT_DIR / "corpus.json").write_text(json.dumps(corpus_json, indent=2))
(OUT_DIR / "golden_dataset.json").write_text(json.dumps(GOLDEN, indent=2))

print(f"Wrote {len(corpus_json)} corpus docs to {OUT_DIR / 'corpus.json'}")
print(f"Wrote {len(GOLDEN)} golden query/role/ground_truth triples to {OUT_DIR / 'golden_dataset.json'}")
