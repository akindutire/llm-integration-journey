"""Function tools the agents can call: scrape the startup page, save the pitch PDF and cold email."""

from typing import Any

import requests
from bs4 import BeautifulSoup

from agents import RunContextWrapper, function_tool

from .structured_outputs import EmailStruct, PitchArtifacts
from .utils import ASSET_DIR, get_artifacts, render_pitch_pdf, write_cold_email_html

# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _extract_title(soup):
    title = soup.find("title")
    return title.text.strip() if title else None


def _extract_meta_description(soup):
    meta = soup.find("meta", attrs={"name": "description"})
    return meta.get("content", "").strip() if meta else None


def _extract_headings(soup):
    headings = {}
    for i in range(1, 7):
        h_tags = soup.find_all(f"h{i}")
        if h_tags:
            headings[f"h{i}"] = [h.text.strip() for h in h_tags if h.text.strip()]
    return headings


def _extract_text_content(soup, limit=500):
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)[:limit]


def scrape_startup_page(url: str) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"url": url, "error": f"Error fetching the page: {e}"}

    soup = BeautifulSoup(response.content, "html.parser")
    return {
        "url": url,
        "status_code": response.status_code,
        "title": _extract_title(soup),
        "meta_description": _extract_meta_description(soup),
        "headings": _extract_headings(soup),
        "text_content": _extract_text_content(soup),
    }


@function_tool
def tell_me_about_startup(url: str) -> dict[str, Any]:
    """This tool gather the about us of the startup"""
    return scrape_startup_page(url)


# ---------------------------------------------------------------------------
# Sales rep: cold email + PDF
# ---------------------------------------------------------------------------


@function_tool
def save_pitch_as_pdf(ctx: RunContextWrapper[PitchArtifacts], pitch_markdown: str) -> dict[str, Any]:
    """save_pitch_as_pdf"""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "pitch.md").write_text(pitch_markdown, encoding="utf-8")

    pdf = render_pitch_pdf(pitch_markdown)
    if pdf is None:
        return {"success": False}
    (ASSET_DIR / "pitch.pdf").write_bytes(pdf)

    artifacts = get_artifacts(ctx)
    if artifacts is not None:
        artifacts.pitch_pdf = pdf
    return {"success": True, "path": str(ASSET_DIR / "pitch.pdf")}


@function_tool
def save_cold_email_as_html(subject: str, body: str) -> dict[str, Any]:
    """save_cold_email_as_html"""
    write_cold_email_html(EmailStruct(subject=subject, body=body))
    return {"success": True, "path": str(ASSET_DIR / "pitch_cold_email.html")}
