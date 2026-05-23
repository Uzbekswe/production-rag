"""
Download SEC EDGAR 10-K filings for our RAG corpus.

Uses the SEC EDGAR public API directly — no edgartools dependency.

Companies: AAPL, MSFT, NVDA, GOOGL, META — 10-K for fiscal years 2023 and 2024.
Output: data/sample_docs/{TICKER}/{TICKER}_10K_{YEAR}.pdf (or .htm if PDF unavailable)

SEC requires a User-Agent header identifying who is making requests.
"""

import time
from pathlib import Path

import httpx

# SEC requires this in every request header
IDENTITY = "Portfolio RAG Project ragdev@portfolio.com"
HEADERS = {"User-Agent": IDENTITY, "Accept-Encoding": "gzip, deflate"}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_docs"

# CIK numbers from SEC EDGAR (zero-padded to 10 digits)
COMPANIES = {
    "AAPL":  ("0000320193",  "Apple Inc"),
    "MSFT":  ("0000789019",  "Microsoft Corp"),
    "NVDA":  ("0001045810",  "NVIDIA Corp"),
    "GOOGL": ("0001652044",  "Alphabet Inc"),
    "META":  ("0001326801",  "Meta Platforms"),
}

FILINGS_PER_COMPANY = 2   # FY2024 + FY2023
REQUEST_DELAY = 0.6        # SEC asks for ≤10 req/s; we do ~1.7 req/s


def get_10k_filings(cik: str) -> list[dict]:
    """Fetch the company's recent 10-K filing metadata from EDGAR submissions API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnums = recent.get("accessionNumber", [])
    primary = recent.get("primaryDocument", [])

    filings = []
    for form, date, acc, doc in zip(forms, dates, accnums, primary):
        if form == "10-K":
            filings.append({"date": date, "accession": acc, "primary_doc": doc})

    # newest first; keep filings from 2023-01-01 onwards
    filings = [f for f in filings if f["date"] >= "2023-01-01"]
    return filings[:FILINGS_PER_COMPANY]


def filing_base_url(cik: str, accession: str) -> str:
    """Build the EDGAR filing folder URL."""
    acc_clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}"


def list_filing_documents(cik: str, accession: str) -> list[dict]:
    """Fetch the filing index to get all document names and types."""
    acc_clean = accession.replace("-", "")
    idx_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=10"

    # Use the machine-readable JSON index
    base = filing_base_url(cik, accession)
    idx_url = f"{base}/{accession}-index.json"
    try:
        resp = httpx.get(idx_url, headers=HEADERS, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        return docs
    except Exception:
        return []


def download_filing(ticker: str, cik: str, filing: dict) -> Path | None:
    """
    Download one 10-K filing. Tries to find a PDF first, falls back to the
    primary HTML document.
    """
    accession = filing["accession"]
    primary   = filing["primary_doc"]
    date      = filing["date"]

    # Derive fiscal year: a 10-K filed in Jan-Jun covers the previous calendar year
    year = int(date[:4])
    month = int(date[5:7])
    fy = year - 1 if month <= 6 else year
    year_label = str(fy)

    company_dir = OUTPUT_DIR / ticker
    company_dir.mkdir(parents=True, exist_ok=True)
    base_url = filing_base_url(cik, accession)

    print(f"  Filing date: {date}  →  FY{fy}  (accession: {accession})")

    # Try to find a PDF in the filing index
    docs = list_filing_documents(cik, accession)
    time.sleep(REQUEST_DELAY)

    pdf_docs = [d for d in docs if d.get("type") == "10-K" and d.get("filename", "").lower().endswith(".pdf")]
    if not pdf_docs:
        # broader search: any PDF attachment
        pdf_docs = [d for d in docs if d.get("filename", "").lower().endswith(".pdf")]

    if pdf_docs:
        filename = pdf_docs[0]["filename"]
        url = f"{base_url}/{filename}"
        dest = company_dir / f"{ticker}_10K_{year_label}.pdf"
        print(f"  → PDF found: {filename}, downloading …")
        try:
            with httpx.stream("GET", url, headers=HEADERS, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                dest.write_bytes(r.read())
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓ Saved {size_kb:,} KB  →  {dest.name}")
            return dest
        except Exception as e:
            print(f"  ! PDF download failed: {e}")

    # Fall back to primary HTML document
    dest = company_dir / f"{ticker}_10K_{year_label}.htm"
    url = f"{base_url}/{primary}"
    print(f"  → No PDF, downloading primary document: {primary}")
    try:
        with httpx.stream("GET", url, headers=HEADERS, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            dest.write_bytes(r.read())
        size_kb = dest.stat().st_size // 1024
        print(f"  ✓ Saved {size_kb:,} KB  →  {dest.name}")
        return dest
    except Exception as e:
        print(f"  ! HTML download failed: {e}")
        return None


def main() -> None:
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Companies: {', '.join(COMPANIES.keys())}")
    print(f"Filing type: 10-K  |  FY2023 + FY2024")
    print("-" * 60)

    results: dict[str, list[str]] = {}

    for ticker, (cik, name) in COMPANIES.items():
        print(f"\n[{ticker}] {name}")
        results[ticker] = []

        try:
            filings = get_10k_filings(cik)
            time.sleep(REQUEST_DELAY)

            if not filings:
                print("  ! No 10-K filings found in date range")
                continue

            for filing in filings:
                dest = download_filing(ticker, cik, filing)
                if dest:
                    results[ticker].append(str(dest))
                time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"  ! Error processing {ticker}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for ticker, files in results.items():
        status = "✓" if files else "✗"
        print(f"  {status} {ticker}: {len(files)} file(s)")
        for f in files:
            p = Path(f)
            size = p.stat().st_size // 1024 if p.exists() else 0
            print(f"      {p.name}  ({size:,} KB)")
        total += len(files)

    expected = len(COMPANIES) * FILINGS_PER_COMPANY
    print(f"\nTotal downloaded: {total} / {expected} expected")
    if total < expected:
        print("Some filings missing — check errors above. Re-running is safe.")


if __name__ == "__main__":
    main()
