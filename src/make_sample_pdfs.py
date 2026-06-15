from __future__ import annotations

from pathlib import Path

from .utils import PDF_DIR


DOCS = {
    "nda_vendor_x.pdf": [
        "Vendor X NDA",
        [
            "Confidential Information means all non-public business, technical, and financial information disclosed by either party.",
            "The term of this agreement is two years from the effective date.",
            "The notice period for termination without cause is thirty (30) days written notice.",
            "No party may assign this agreement without prior written consent.",
            "This agreement is governed by the laws of the State of Delaware.",
        ],
    ],
    "msa_vendor_y.pdf": [
        "Vendor Y MSA",
        [
            "Services are provided on a subscription basis and invoiced monthly in advance.",
            "The limitation of liability shall not exceed INR 1 crore in aggregate.",
            "The vendor will maintain uptime of 99.5 percent excluding scheduled maintenance.",
            "Customer may terminate for material breach with a fifteen (15) day cure period.",
            "Disputes will be resolved by arbitration in Mumbai.",
        ],
    ],
    "policy_retention.pdf": [
        "Records Retention Policy",
        [
            "Employee records must be retained for seven years after employment ends.",
            "Finance records must be retained for eight years.",
            "Legal hold notices override ordinary deletion schedules.",
            "Access to archived files requires approval from the compliance team.",
            "Policy exceptions must be approved in writing by the General Counsel.",
        ],
    ],
}


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_content(title: str, paragraphs: list[str]) -> bytes:
    lines = [f"BT /F1 14 Tf 72 760 Td ({_escape_pdf_text(title)}) Tj"]
    y = 735
    for para in paragraphs:
        for line in wrap_text(para, 88):
            lines.append(f"BT /F1 11 Tf 72 {y} Td ({_escape_pdf_text(line)}) Tj")
            y -= 16
    lines.append("ET")
    return "\n".join(lines).encode("latin-1")


def _make_pdf_bytes(title: str, paragraphs: list[str]) -> bytes:
    content = _build_content(title, paragraphs)
    objs: list[bytes] = []

    def add_obj(body: bytes) -> None:
        objs.append(body)

    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    add_obj(page)
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_obj(
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs)+1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
    )
    return bytes(out)


def wrap_text(text: str, width: int) -> list[str]:
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


def make_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    path.write_bytes(_make_pdf_bytes(title, paragraphs))


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, paragraphs) in DOCS.items():
        make_pdf(PDF_DIR / filename, title, paragraphs)


if __name__ == "__main__":
    main()
