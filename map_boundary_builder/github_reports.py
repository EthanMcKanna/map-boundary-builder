from __future__ import annotations

import base64
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_REPORT_REPOSITORY = "EthanMcKanna/map-boundary-builder"
DEFAULT_REPORT_BRANCH = "debug-reports"
MAX_REPORT_IMAGE_BYTES = 8 * 1024 * 1024


class GithubReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationReport:
    filename: str
    image_bytes: bytes
    error: str
    issue_type: str = "Generation failed"
    generation_status: str = "failed"
    user_note: str | None = None
    run_id: str | None = None
    events: list[dict[str, Any]] | None = None
    user_agent: str | None = None
    page_url: str | None = None
    settings: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None


FailureReport = GenerationReport


def create_failure_issue(report: GenerationReport) -> dict[str, Any]:
    token = github_token()
    if not token:
        raise GithubReportError("GitHub reporting is not configured on this deployment.")

    repository = os.environ.get("GITHUB_REPORT_REPOSITORY", DEFAULT_REPORT_REPOSITORY).strip()
    branch = os.environ.get("GITHUB_REPORT_BRANCH", DEFAULT_REPORT_BRANCH).strip()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise GithubReportError("GitHub report repository is invalid.")
    if not branch:
        raise GithubReportError("GitHub report branch is invalid.")

    if not report.image_bytes:
        raise GithubReportError("Report image is empty.")
    if len(report.image_bytes) > MAX_REPORT_IMAGE_BYTES:
        raise GithubReportError("Report image is larger than 8 MB.")

    report_id = f"{int(time.time())}-{secrets.token_hex(4)}"
    ext = safe_report_extension(report.filename)
    artifact_path = f"debug-reports/{time.strftime('%Y-%m-%d')}/{report_id}/input{ext}"
    ensure_branch(repository, branch, token)
    upload = upload_report_image(repository, branch, artifact_path, report, token)
    image_url = upload.get("download_url") or raw_github_url(repository, branch, artifact_path)
    issue = create_issue(repository, branch, artifact_path, image_url, report, token)
    return {
        "issue_url": issue.get("html_url"),
        "issue_number": issue.get("number"),
        "image_url": image_url,
    }


def github_token() -> str | None:
    for name in ("GITHUB_REPORT_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def ensure_branch(repository: str, branch: str, token: str) -> None:
    encoded_branch = quote(branch, safe="")
    try:
        github_json(f"/repos/{repository}/git/ref/heads/{encoded_branch}", token=token)
        return
    except GithubReportError as exc:
        if "404" not in str(exc):
            raise

    repo = github_json(f"/repos/{repository}", token=token)
    default_branch = repo.get("default_branch") or "main"
    default_ref = github_json(
        f"/repos/{repository}/git/ref/heads/{quote(str(default_branch), safe='')}",
        token=token,
    )
    sha = default_ref.get("object", {}).get("sha")
    if not sha:
        raise GithubReportError("Could not resolve the repository default branch.")
    github_json(
        f"/repos/{repository}/git/refs",
        token=token,
        method="POST",
        payload={"ref": f"refs/heads/{branch}", "sha": sha},
    )


def upload_report_image(
    repository: str,
    branch: str,
    artifact_path: str,
    report: GenerationReport,
    token: str,
) -> dict[str, Any]:
    content = base64.b64encode(report.image_bytes).decode("ascii")
    payload = {
        "message": "Add Boundary Builder report image",
        "content": content,
        "branch": branch,
    }
    return github_json(
        f"/repos/{repository}/contents/{quote(artifact_path, safe='/')}",
        token=token,
        method="PUT",
        payload=payload,
    ).get("content", {})


def create_issue(
    repository: str,
    branch: str,
    artifact_path: str,
    image_url: str,
    report: GenerationReport,
    token: str,
) -> dict[str, Any]:
    title = "{} report: {}".format(
        sanitize_title(report.issue_type),
        Path(report.filename).name or "uploaded image",
    )
    body = issue_body(branch, artifact_path, image_url, report)
    return github_json(
        f"/repos/{repository}/issues",
        token=token,
        method="POST",
        payload={"title": title[:256], "body": body},
    )


def issue_body(branch: str, artifact_path: str, image_url: str, report: GenerationReport) -> str:
    events = json.dumps(report.events or [], indent=2, default=str)
    settings = json.dumps(report.settings or {}, indent=2, default=str)
    summary = json.dumps(report.summary or {}, indent=2, default=str)
    profile = json.dumps(report.profile or {}, indent=2, default=str)
    return "\n".join(
        [
            "A user reported a Boundary Builder generation from the app.",
            "",
            "**Public image notice:** the uploaded screenshot is intentionally stored in this public GitHub repository for debugging.",
            "",
            f"![Reported map screenshot]({image_url})",
            "",
            f"- Issue type: `{report.issue_type or 'Unspecified'}`",
            f"- Generation status: `{report.generation_status or 'unknown'}`",
            f"- Original filename: `{Path(report.filename).name or 'uploaded-image'}`",
            f"- Run ID: `{report.run_id or 'not available'}`",
            f"- Debug branch: `{branch}`",
            f"- Image path: `{artifact_path}`",
            f"- Page URL: `{report.page_url or 'not provided'}`",
            f"- User agent: `{report.user_agent or 'not provided'}`",
            "",
            "## User Note",
            "",
            (report.user_note or "No additional note provided.")[:4000],
            "",
            "## Error Or Status",
            "",
            "```text",
            (report.error or "No error message provided.")[:4000],
            "```",
            "",
            "## Generation Summary",
            "",
            "```json",
            summary[:12000],
            "```",
            "",
            "## Settings",
            "",
            "```json",
            settings[:8000],
            "```",
            "",
            "## Runtime Profile",
            "",
            "```json",
            profile[:8000],
            "```",
            "",
            "## Recent Events",
            "",
            "```json",
            events[:16000],
            "```",
        ]
    )


def github_json(
    path: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "map-boundary-builder-reporting",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        f"https://api.github.com{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise GithubReportError(f"GitHub API request failed with {exc.code}: {detail}") from exc
    except URLError as exc:
        raise GithubReportError(f"GitHub API request failed: {exc.reason}") from exc
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def safe_report_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".avif", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return ext
    return ".png"


def sanitize_title(value: str) -> str:
    title = re.sub(r"\s+", " ", str(value or "Generation issue")).strip()
    return title[:80] or "Generation issue"


def raw_github_url(repository: str, branch: str, artifact_path: str) -> str:
    return "https://raw.githubusercontent.com/{}/{}/{}".format(
        repository,
        quote(branch, safe=""),
        quote(artifact_path, safe="/"),
    )
