from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random

from .utils import PDF_DIR


@dataclass
class DocMeta:
    filename: str
    title: str
    doc_type: str
    customer: str
    vendor: str
    effective_date: str
    governing_law: str
    dispute_resolution: str
    venue: str
    term_months: int
    notice_days: int
    liability_cap: str
    payment_days: int
    service_level: str
    confidentiality_years: int
    data_retention_years: int
    edge_cases: dict[str, bool] = field(default_factory=dict)


def make_docs() -> list[DocMeta]:
    return [
        DocMeta("nda_vendor_x.pdf", "Mutual Non-Disclosure Agreement", "NDA", "Acme Holdings", "Vendor X", "2025-01-15", "Delaware", "arbitration", "Wilmington", 24, 30, "USD 250,000", 15, "99.5%", 3, 7, {"split_clause": True, "duplicate_definition": True, "cross_reference": True}),
        DocMeta("msa_vendor_y.pdf", "Master Services Agreement", "MSA", "Northstar Retail", "Vendor Y", "2025-02-01", "New York", "court litigation", "Manhattan", 36, 45, "INR 1 crore", 30, "99.7%", 5, 8, {"split_clause": True, "cross_reference": True, "table": True, "conflicting_exhibit": True}),
        DocMeta("policy_retention.pdf", "Records Retention and Data Handling Policy", "POLICY", "Internal", "Internal", "2024-12-01", "Singapore", "arbitration", "Singapore", 12, 10, "N/A", 0, "98.0%", 2, 7, {"bullet_list": True, "cross_reference": True, "table": True, "annex": True}),
        DocMeta("dpa_vendor_x.pdf", "Data Processing Addendum", "DPA", "Acme Holdings", "Vendor X", "2025-01-15", "Ireland", "arbitration", "Dublin", 24, 14, "USD 500,000", 15, "99.9%", 4, 6, {"split_clause": True, "cross_reference": True, "table": True, "annex": True, "bullet_list": True}),
        DocMeta("saas_platform.pdf", "SaaS Subscription Agreement", "SAAS", "BluePeak Labs", "Vendor Z", "2025-03-10", "California", "court litigation", "San Francisco", 12, 30, "USD 100,000", 30, "99.5%", 3, 5, {"split_clause": True, "cross_reference": True, "table": True, "conflicting_exhibit": True}),
    ]


def build_sections(meta: DocMeta) -> list[tuple[str, list[str]]]:
    return [
        ("Recitals", [
            f"{meta.customer} desires to obtain services from {meta.vendor} and both parties wish to record the commercial and legal terms in a single integrated agreement.",
            "The parties acknowledge that this Agreement is intended to govern the relationship between them and supersedes prior discussions, drafts, term sheets, and email summaries relating to the same subject matter.",
            "The parties further acknowledge that specific obligations may be supplemented by exhibits, schedules, and annexes, but only to the extent that those documents are expressly incorporated by reference.",
            "For clarity, headings are for convenience only and do not affect interpretation, while any examples are illustrative and not exhaustive unless expressly stated otherwise.",
        ]),
        ("1. Definitions", [
            f'"Confidential Information" means non-public business, technical, financial, operational, or security information disclosed by {meta.customer} or {meta.vendor}, whether in written, oral, visual, electronic, or other form, and includes derivatives, analyses, and summaries derived from that information.',
            f'"Services" means the work and deliverables described in Exhibit A, together with any support, advisory, configuration, testing, training, and onboarding services that are reasonably necessary to complete those deliverables, subject always to Section 4.2 and the service levels in Section 6.',
            f'"Effective Date" means {meta.effective_date}, provided that if execution occurs later than the stated date, then the effective date is the date of the last signature unless the parties agree otherwise in writing.',
            f'"Affiliate" means any entity that directly or indirectly controls, is controlled by, or is under common control with a party, and control means ownership of more than fifty percent of the voting interests or the practical ability to direct management decisions.',
        ]),
        ("2. Scope of Services", [
            f"{meta.vendor} will provide implementation and support services to {meta.customer} in accordance with this Agreement, using commercially reasonable efforts, appropriately qualified personnel, and generally accepted industry practices for projects of comparable size and complexity.",
            "The parties acknowledge that service descriptions in Exhibit A supplement, but do not replace, the main body of this Agreement, and that any ambiguity will be resolved in a manner that preserves the overall commercial intent of the parties as expressed in the recitals and the operative clauses.",
            "Any work outside the defined scope requires a written change order signed by both parties, and the change order must describe the commercial impact, timeline impact, and any dependency on customer-provided materials or third-party systems.",
            "Unless expressly agreed in writing, the vendor is not responsible for delays caused by incomplete inputs, third-party outages, or changes in law that require modifications to the service design.",
        ]),
        ("3. Term and Termination", [
            f"This Agreement begins on the Effective Date and continues for {meta.term_months} months unless terminated earlier under this Section, and any automatic renewal shall be for successive twelve-month periods unless either party gives the required notice of non-renewal.",
            f"Either party may terminate for convenience upon {meta.notice_days} days' prior written notice, provided that such notice must specify whether the termination relates to the entire Agreement or only to a defined statement of work or exhibit.",
            "Either party may terminate immediately for material breach if the breach is not cured within 15 days after written notice, except that breaches involving confidentiality, data security, or non-payment may have shorter or longer cure periods if a mandatory law requires it.",
            "Termination does not affect accrued payment obligations, confidentiality obligations, or any rights that by their nature survive termination, including audit rights, indemnity claims, and obligations to return or destroy data.",
        ]),
        ("4. Fees and Payment", [
            f"Invoices are payable within {meta.payment_days} days of receipt, excluding disputed amounts that are raised in good faith and documented with sufficient detail to permit the receiving party to investigate the dispute.",
            "Late payments may accrue interest at the maximum rate permitted by law, and the non-defaulting party may suspend undisputed services only after giving reasonable prior notice and a final opportunity to cure.",
            "Fees do not include taxes, withholding, or bank transfer charges unless expressly stated in Exhibit B, and any withholding tax must be supported by a valid certificate or other legally required documentation.",
            "If the parties agree to milestone-based billing, each milestone invoice must reference the applicable milestone, acceptance criteria, and any customer sign-off that supports the invoice amount.",
        ]),
        ("5. Confidentiality", [
            f"Each party shall protect Confidential Information with at least reasonable care and use it only for the purposes of this Agreement, and may disclose Confidential Information only to employees, contractors, or advisors who need to know the information and are bound by written confidentiality obligations no less protective than those in this Agreement.",
            f"Confidentiality obligations survive for {meta.confidentiality_years} years after termination, except for trade secrets which remain protected as long as allowed by law, and except for records that must be preserved under a legal hold or records-retention law.",
            "A disclosure required by law is permitted only to the extent legally required and after prompt notice to the disclosing party where lawful, so that the disclosing party may seek protective relief or narrow the scope of disclosure.",
            "Information that is publicly available through no breach of this Agreement is not Confidential Information, but the burden of showing such public availability rests on the receiving party.",
        ]),
        ("6. Service Levels", [
            f"{meta.vendor} will maintain uptime of {meta.service_level}, excluding scheduled maintenance, force majeure, and outages caused by customer systems or customer-controlled integrations.",
            "If uptime falls below the committed level in two consecutive months, the customer may request service credits described in Exhibit C, subject to any cap on credits stated there and subject also to timely notice of the SLA breach.",
            "Support response times are measured during business hours in the customer's primary operating region, and any response time calculation excludes delays caused by missing information or by a request that is outside the supported scope.",
            "Service measurements will be calculated from the monitoring logs retained by the vendor, except where the customer produces materially contradictory evidence from its own monitoring tools.",
        ]),
        ("7. Indemnity and Liability", [
            f"Subject to the exclusions in this Section, each party's aggregate liability shall not exceed {meta.liability_cap}, whether the claim sounds in contract, tort, statute, or otherwise.",
            "This cap does not apply to fraud, willful misconduct, confidentiality breaches, or indemnification obligations, and in some cases may also exclude payment obligations or misuse of intellectual property if required by law.",
            "The indemnifying party will defend, indemnify, and hold harmless the other party against third-party claims arising from IP infringement or gross negligence, and the indemnified party must provide prompt notice and reasonable cooperation.",
            "The indemnifying party may not settle a claim in a way that imposes an admission of fault or an obligation on the indemnified party without the indemnified party's prior written consent.",
        ]),
        ("8. Data Protection", [
            f"{meta.vendor} shall implement appropriate technical and organizational measures to protect personal data and shall retain such data only for {meta.data_retention_years} years unless a longer period is required by law or by a valid legal hold.",
            "A data breach must be reported without undue delay and in any event within 72 hours after confirmation, unless restricted by law enforcement or another lawful confidentiality obligation.",
            "Subprocessors may be used only with equivalent obligations and documented approval from the customer, and the vendor remains responsible for their acts and omissions as if those acts and omissions were its own.",
            "Upon termination, the vendor will return or delete personal data within a commercially reasonable period, subject to retention required by law.",
        ]),
        ("9. Assignment; Change of Control", [
            "Neither party may assign this Agreement without the other party's prior written consent, except to an affiliate or in connection with a merger or sale of substantially all assets, provided that the assignee assumes all obligations in writing.",
            "Consent will not be unreasonably withheld, conditioned, or delayed, and any attempted assignment in violation of this Section is void.",
            "A change of control will be treated as an assignment only if the relevant transaction results in a transfer of the Agreement or the practical ability to control performance under the Agreement.",
        ]),
        ("10. Governing Law and Dispute Resolution", [
            f"This Agreement is governed by the laws of {meta.governing_law}, without regard to conflict-of-law principles that would apply the law of another jurisdiction.",
            f"Any dispute shall be resolved by {meta.dispute_resolution} seated in {meta.venue}, except that either party may seek injunctive relief in a court of competent jurisdiction to protect confidentiality, data security, or intellectual property rights.",
            "The prevailing party in a dispute may recover reasonable legal fees and costs to the extent permitted by law or the chosen dispute forum.",
        ]),
    ]


def build_exhibits(meta: DocMeta) -> list[tuple[str, list[str]]]:
    exhibits: list[tuple[str, list[str]]] = [
        ("Exhibit A. Statement of Work", [
            f"{meta.vendor} will provide onboarding, implementation, migration, and support services for {meta.customer}, including reasonable assistance with testing, knowledge transfer, and go-live planning.",
            "Deliverables include configuration, testing support, go-live assistance, hypercare support, and a post-launch knowledge transfer session.",
            "Any assumptions stated here are incorporated by reference into the main Agreement, including assumptions about customer-side dependencies, data quality, and access to required systems.",
            "If a deliverable requires customer review, acceptance is deemed granted unless the customer provides written rejection specifying a material non-conformity within five business days.",
        ]),
        ("Exhibit B. Commercial Terms", [
            "This table sets forth the key commercial assumptions for the Agreement.",
            f"Fees | Payment Terms | Renewal | Currency",
            f"Fixed monthly fee | Net {meta.payment_days} | Auto-renew | INR",
            "Usage-based fees require written approval before billing.",
            "Any discount applies only to the expressly identified service period and does not survive a material change in scope unless the parties sign a written amendment.",
        ]),
    ]
    if meta.edge_cases.get("conflicting_exhibit"):
        exhibits.append(("Exhibit C. Special Terms", [
            "For this exhibit only, the notice period is 60 days. This carve-out applies only to pilot services described in Exhibit A and only during the pilot window defined by the parties.",
            "Any conflict between this Exhibit and the main Agreement is resolved in favor of the main Agreement unless expressly stated otherwise, and a later signed change order supersedes both only to the extent of the specific change it describes.",
            "The parties agree that special terms should be interpreted narrowly so they do not unintentionally override the main risk allocation in the Agreement.",
        ]))
    if meta.edge_cases.get("annex"):
        exhibits.append(("Annex 1. Retention Schedule", [
            "Employee records: 7 years.",
            "Finance records: 8 years.",
            "Security logs: 2 years.",
            "Legal hold notices override the ordinary retention schedule.",
            "The vendor may retain archival copies solely for compliance, audit, or backup purposes if access is restricted and deletion would be impracticable.",
        ]))
    return exhibits


def apply_style(line: str) -> tuple[str, int, int, int]:
    s = line.strip()
    if not s:
        return ("F1", 11, 72, 12)
    if s.startswith("Footer:"):
        return ("F4", 8, 72, 9)
    if s.startswith("Table of Contents") or s in {"Recitals", "Signature Blocks", "Schedule of Key Terms", "Operational Summary Table", "Policy Exceptions", "Split Clause Example"} or s.startswith("Exhibit") or s.startswith("Annex"):
        return ("F2", 13, 72, 16)
    if s.startswith("Page ") or s.startswith("Parties:") or s.startswith("Confidential"):
        return ("F2", 10, 72, 12)
    if s.startswith("-"):
        return ("F3", 10, 84, 12)
    if " | " in s:
        return ("F3", 10, 76, 11)
    if s[0].isdigit() and "." in s[:4]:
        return ("F1", 10, 78, 12)
    if "________________" in s:
        return ("F4", 11, 72, 12)
    return ("F1", 11, 72, 12)


def mix_clause_styles(clauses: list[str], rng: random.Random) -> list[str]:
    variants = [
        lambda c: c,
        lambda c: f"For the avoidance of doubt, {c[0].lower()}{c[1:]}" if c else c,
        lambda c: f"Notwithstanding anything to the contrary in this Agreement, {c[0].lower()}{c[1:]}" if c else c,
        lambda c: f"Except as otherwise expressly stated in this Agreement, {c[0].lower()}{c[1:]}" if c else c,
        lambda c: f"{c} (including, without limitation, related deliverables, support obligations, and reasonable transition assistance.)",
    ]
    mixed: list[str] = []
    for clause in clauses:
        mixed.append(rng.choice(variants)(clause))
        if rng.random() < 0.25:
            mixed.append("Provided, however, that the foregoing is subject to Section 7 and any mandatory law applicable to the parties.")
        if rng.random() < 0.15:
            mixed.append("Any similar obligation implied by course of dealing or prior conduct is excluded unless incorporated by a signed amendment.")
    return mixed


def pack_blocks_into_pages(blocks: list[tuple[str, list[str]]], target_body_lines: int) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            pages.append(current)
            current = []
            current_len = 0

    for title, block in blocks:
        rendered = [title]
        rendered.extend([f"{i}. {line}" if not line.startswith("-") and not line.startswith("Clause |") else line for i, line in enumerate(block, start=1)])
        rendered.append("")
        if current_len + len(rendered) > target_body_lines and current:
            flush()
        current.extend(rendered)
        current_len += len(rendered)
    flush()
    return pages


def build_long_document(meta: DocMeta) -> list[list[str]]:
    blocks: list[tuple[str, list[str]]] = []
    cover = [
        f"{meta.title}",
        f"{meta.doc_type} between {meta.customer} and {meta.vendor}",
        f"Effective Date: {meta.effective_date}",
        "This Agreement is made as of the Effective Date above and is intended to be a binding legal instrument.",
        "This document includes recitals, operative clauses, exhibits, annexes, a commercial summary, and signature blocks.",
        "",
        "Key commercial snapshot:",
        f"- Notice period: {meta.notice_days} days",
        f"- Liability cap: {meta.liability_cap}",
        f"- Uptime commitment: {meta.service_level}",
        f"- Payment term: {meta.payment_days} days",
        f"- Confidentiality survival: {meta.confidentiality_years} years",
        f"- Data retention: {meta.data_retention_years} years",
    ]
    toc = [
        "Table of Contents",
        "1. Recitals",
        "2. Definitions",
        "3. Scope of Services",
        "4. Term and Termination",
        "5. Fees and Payment",
        "6. Confidentiality",
        "7. Service Levels",
        "8. Indemnity and Liability",
        "9. Data Protection",
        "10. Assignment; Change of Control",
        "11. Governing Law and Dispute Resolution",
        "Exhibits A-C",
        "Annexes and Signature Blocks",
    ]
    blocks.append(("Cover", cover))
    blocks.append(("TOC", toc))
    for title, clauses in build_sections(meta):
        blocks.append((title, clauses))
    for title, clauses in build_exhibits(meta):
        blocks.append((title, clauses))
    blocks.append(("Schedule of Key Terms", [
        f"Notice Period: {meta.notice_days} days",
        f"Liability Cap: {meta.liability_cap}",
        f"Uptime Commitment: {meta.service_level}",
        f"Payment Term: {meta.payment_days} days",
        f"Confidentiality Survival: {meta.confidentiality_years} years",
        f"Data Retention: {meta.data_retention_years} years",
        f"Governing Law: {meta.governing_law}",
        f"Dispute Resolution: {meta.dispute_resolution} in {meta.venue}",
    ]))
    if meta.edge_cases.get("table"):
        blocks.append(("Operational Summary Table", [
            "Clause | Value | Notes",
            f"Termination Notice | {meta.notice_days} days | written notice required",
            f"Liability Cap | {meta.liability_cap} | subject to carve-outs",
            f"Uptime | {meta.service_level} | excludes maintenance",
            f"Retention | {meta.data_retention_years} years | legal hold overrides",
        ]))
    if meta.edge_cases.get("bullet_list"):
        blocks.append(("Policy Exceptions", [
            "- legal hold notices override deletion schedules",
            "- approved security exceptions must be documented",
            "- retention changes require compliance sign-off",
            "- emergency access must be logged and reviewed",
            "- third-party disclosures must follow approval procedures",
        ]))
    if meta.edge_cases.get("split_clause"):
        blocks.append(("Split Clause Example", [
            "The parties agree that the following clause continues across pages and must be read together.",
            "The vendor must maintain confidentiality, preserve records, provide prompt notice of material service interruptions, and coordinate with the customer before any material operational change, subject to any shorter notice period required by law or expressly agreed in writing.",
            "This obligation survives termination and applies to subcontractors and affiliates where relevant.",
        ]))
    blocks.append(("Signature Blocks", [
        f"Authorized Signatory for {meta.customer}: ____________________",
        f"Name: ____________________   Title: ____________________",
        "",
        f"Authorized Signatory for {meta.vendor}: ____________________",
        f"Name: ____________________   Title: ____________________",
        "",
        "Each signatory represents that they have authority to bind their respective party.",
    ]))

    page_bodies = pack_blocks_into_pages(blocks, target_body_lines=24)
    pages: list[list[str]] = []
    total = len(page_bodies) + 2
    for i, body in enumerate(page_bodies, start=1):
        header = [
            f"{meta.title} - Page {i}",
            f"{meta.doc_type} | {meta.customer} v. {meta.vendor}",
            f"Page {i} of {total}",
            "Confidential and subject to the terms of the agreement.",
            "",
        ]
        footer = ["", f"Footer: {meta.title} | {meta.effective_date} | {meta.governing_law}"]
        pages.append(header + body + footer)
    return pages


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines, line = [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines


def _make_pdf_bytes(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []
    page_count = len(pages)
    pages_id = 2
    first_page_id = 3
    font1_id = 3 + page_count
    font2_id = font1_id + 1
    font3_id = font1_id + 2
    font4_id = font1_id + 3
    first_content_id = font4_id + 1

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{i} 0 R" for i in range(first_page_id, first_page_id + page_count))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    content_blobs: list[bytes] = []

    for page_lines in pages:
        content_lines = []
        y = 760
        for line in page_lines:
            if not line:
                y -= 8
                continue
            font, size, x, step = apply_style(line)
            for chunk in _wrap_text(line, 90) or [""]:
                content_lines.append(f"BT /{font} {size} Tf {x} {y} Td ({_escape_pdf_text(chunk)}) Tj")
                y -= step
        content_blobs.append("\n".join(content_lines).encode("latin-1"))

    for i in range(page_count):
        content_id = first_content_id + i
        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R /F3 {font3_id} 0 R /F4 {font4_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects.append(page_obj)

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Italic >>")
    for content in content_blobs:
        objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(out)


def make_pdf(path: Path, pages: list[list[str]]) -> None:
    path.write_bytes(_make_pdf_bytes(pages))


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for meta in make_docs():
        make_pdf(PDF_DIR / meta.filename, build_long_document(meta))


if __name__ == "__main__":
    main()
