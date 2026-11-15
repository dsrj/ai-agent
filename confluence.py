"""Explicit, space-scoped Confluence Cloud import; no token is stored in Azure or Terraform."""

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from uuid import UUID

import httpx
from azure.identity import DefaultAzureCredential

from cloudops_hub.config import get_settings
from cloudops_hub.confluence_images import ConfluenceImages
from cloudops_hub.ingest import ingest
from cloudops_hub.retrieval import create_openai_client

logger = logging.getLogger(__name__)


class PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0
        self.table_header = False
        self.link: str | None = None
        self.in_cell = False
        self.macros: list[bool] = []
        self.parameter_hidden = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        elif self.hidden:
            return
        elif tag == "ac:structured-macro":
            code = dict(attrs).get("ac:name") == "code"
            self.macros.append(code)
            if code:
                self.parts.append("\n\n```\n")
        elif tag == "ac:parameter" and any(self.macros):
            self.parameter_hidden = True
        elif re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "pre":
            self.parts.append("\n\n```\n")
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.parts.append(" | ")
            self.table_header = self.table_header or tag == "th"
        elif tag == "a":
            href = dict(attrs).get("href", "")
            self.link = href if href.startswith("https://") else None
            if self.link:
                self.parts.append("[")
        elif tag in {"p", "br", "div", "li", "tr"}:
            if self.in_cell:
                self.parts.append(" ")
                return
            self.parts.append("\n\n" if tag in {"p", "div"} else "\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif self.hidden:
            return
        elif tag == "ac:parameter":
            self.parameter_hidden = False
        elif tag == "ac:structured-macro" and self.macros:
            if self.macros.pop():
                self.parts.append("\n```\n\n")
        elif tag in {"td", "th"}:
            self.in_cell = False
        elif tag == "pre":
            self.parts.append("\n```\n\n")
        elif tag == "a" and self.link:
            self.parts.append(f"]({self.link})")
            self.link = None
        elif tag == "tr":
            self.parts.append(" |\n")
            if self.table_header:
                row = "".join(self.parts).splitlines()[-1]
                self.parts.append("|" + " --- |" * (row.count("|") - 1) + "\n")
                self.table_header = False
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append(" " if self.in_cell else "\n\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden and not self.parameter_hidden:
            self.parts.append(data.replace("|", "\\|").replace("\n", " ") if self.in_cell else data)

    def unknown_decl(self, data: str) -> None:
        if data.startswith("CDATA["):
            self.handle_data(data[6:])


def validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".atlassian.net")
        or parsed.port not in (None, 443)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/wiki"
    ):
        raise ValueError("CONFLUENCE_BASE_URL must be https://<site>.atlassian.net/wiki")
    return value.rstrip("/")


def export_space(
    client: httpx.Client,
    base_url: str,
    space_key: str,
    destination: Path,
    *,
    cloud_id: str,
    root_page_id: str | None = None,
    max_pages: int = 100,
    images: Callable[[dict], str] | None = None,
) -> dict[str, dict[str, str]]:
    base_url = validate_base_url(base_url)
    try:
        cloud_id = str(UUID(cloud_id))
    except ValueError:
        raise ValueError("CONFLUENCE_CLOUD_ID must be the site's UUID") from None
    if not re.fullmatch(r"[A-Za-z0-9_~-]+", space_key):
        raise ValueError("CONFLUENCE_SPACE_KEY must be a single space key")
    if max_pages < 1:
        raise ValueError("Confluence page limit must be positive")
    if root_page_id is not None and not re.fullmatch(r"[0-9]+", root_page_id):
        raise ValueError("CONFLUENCE_ROOT_PAGE_ID must be a numeric page ID")
    api_url = f"https://api.atlassian.com/ex/confluence/{cloud_id}/wiki/api/v2"
    response = client.get(
        f"{api_url}/spaces", params={"keys": space_key, "status": "current", "limit": 2}
    )
    check_response(response)
    spaces = response.json()["results"]
    if (
        not isinstance(spaces, list)
        or len(spaces) != 1
        or spaces[0].get("key") != space_key
        or not str(spaces[0].get("id", "")).isdigit()
    ):
        raise ValueError("Confluence did not return exactly the approved space")
    space_id = str(spaces[0]["id"])
    metadata = {}
    cursor = None
    cursors = set()
    seen = set()
    hierarchy: dict[str, str | None] = {}
    listing_limit = 10000 if root_page_id is not None else max_pages
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        params: dict[str, str | int] = {
            "status": "current",
            "limit": min(25, listing_limit),
        }
        if root_page_id is None:
            params["body-format"] = "storage"
        if cursor is not None:
            params["cursor"] = cursor
        response = client.get(
            f"{api_url}/spaces/{space_id}/pages",
            params=params,
        )
        check_response(response)
        payload = response.json()
        pages = payload["results"]
        if not isinstance(pages, list):
            raise ValueError("Confluence returned an invalid page list")
        for page in pages:
            if str(page["spaceId"]) != space_id or page["status"] != "current":
                raise ValueError("Confluence returned a page outside the approved space/status")
            page_id = str(page["id"])
            if not page_id.isdigit() or page_id in seen:
                raise ValueError("Confluence returned an invalid or repeated page ID")
            seen.add(page_id)
            if len(seen) > listing_limit:
                raise RuntimeError("Confluence page limit exceeded; no index changes made")
            if root_page_id is not None:
                hierarchy[page_id] = page["parentId"]
                continue
            exported = write_page(page, base_url, destination, images=images)
            if exported is not None:
                metadata[f"{page_id}.txt"] = exported
        next_url = payload.get("_links", {}).get("next") or response.links.get("next", {}).get(
            "url"
        )
        if not next_url:
            break
        if not pages:
            raise ValueError("Confluence pagination did not advance")
        if len(seen) >= listing_limit:
            raise RuntimeError("Confluence page limit exceeded; no index changes made")
        # Extract only the opaque cursor; never send credentials to a returned URL.
        next_cursors = parse_qs(urlparse(next_url).query).get("cursor", [])
        if len(next_cursors) != 1 or next_cursors[0] in cursors:
            raise ValueError("Confluence returned an invalid or repeated pagination cursor")
        cursor = next_cursors[0]
        cursors.add(cursor)
    if root_page_id is not None:
        if root_page_id not in hierarchy:
            raise ValueError("Approved root page is not a readable published page in this space")
        children: dict[str, list[str]] = {}
        for page_id, parent_id in hierarchy.items():
            if parent_id is not None:
                children.setdefault(parent_id, []).append(page_id)
        # Pages inside Confluence folders have a folder parent, which the page listing
        # does not include. Resolve each such parent once through a child's ancestors.
        for parent_id in sorted(set(children) - set(hierarchy) - {hierarchy[root_page_id]}):
            if root_page_id in page_ancestors(client, api_url, children[parent_id][0]):
                children.setdefault(root_page_id, []).extend(children[parent_id])
        pending = [root_page_id]
        selected = set()
        while pending:
            page_id = pending.pop()
            if page_id in selected:
                raise ValueError("Confluence returned a cyclic page hierarchy")
            selected.add(page_id)
            if len(selected) > max_pages:
                raise RuntimeError("Confluence page limit exceeded; no index changes made")
            pending.extend(children.get(page_id, []))
        for page_id in sorted(selected):
            response = client.get(
                f"{api_url}/pages/{page_id}",
                params={"status": "current", "body-format": "storage"},
            )
            check_response(response)
            page = response.json()
            if (
                str(page["id"]) != page_id
                or str(page["spaceId"]) != space_id
                or page["status"] != "current"
                or page.get("parentId") != hierarchy[page_id]
            ):
                raise ValueError("Confluence page changed outside the approved hierarchy")
            exported = write_page(page, base_url, destination, images=images)
            if exported is not None:
                metadata[f"{page_id}.txt"] = exported
    if not metadata:
        raise RuntimeError(
            "Confluence returned no pages with importable content; "
            "existing indexed content is unchanged"
        )
    return metadata


ANCESTOR_LIMIT = 250


def page_ancestors(client: httpx.Client, api_url: str, page_id: str) -> set[str]:
    response = client.get(f"{api_url}/pages/{page_id}/ancestors", params={"limit": ANCESTOR_LIMIT})
    check_response(response)
    results = response.json().get("results")
    if not isinstance(results, list) or len(results) >= ANCESTOR_LIMIT:
        raise ValueError("Confluence returned an invalid or too deep ancestor list")
    ancestors = set()
    for item in results:
        ancestor_id = str(item.get("id", "")) if isinstance(item, dict) else ""
        if not ancestor_id.isdigit():
            raise ValueError("Confluence returned an invalid ancestor ID")
        ancestors.add(ancestor_id)
    return ancestors


def write_page(
    page: dict, base_url: str, destination: Path, *, images: Callable[[dict], str] | None = None
) -> dict[str, str] | None:
    page_id = str(page["id"])
    parser = PageText()
    parser.feed(page["body"]["storage"]["value"])
    content = "".join(parser.parts).strip()
    webui = page["_links"]["webui"]
    parsed_webui = urlparse(webui)
    if parsed_webui.scheme or parsed_webui.netloc or webui.startswith("/wiki/"):
        page_url = urljoin(base_url + "/", webui)
    else:
        page_url = urljoin(base_url + "/", webui.lstrip("/"))
    if urlparse(page_url).netloc != urlparse(base_url).netloc:
        raise ValueError("Confluence returned a page URL on a different host")
    if urlparse(page_url).scheme != "https":
        raise ValueError("Confluence returned a non-HTTPS page URL")
    if images:
        image_content = images(page)
        if image_content:
            content = f"{content}\n\n{image_content}".strip()
    if not content:
        if images is None:
            logger.warning(
                "Skipping Confluence page %s: no importable text in text-only mode", page_id
            )
            return None
        raise ValueError(f"Confluence page {page_id} has no importable text or images")
    (destination / f"{page_id}.txt").write_text(content, encoding="utf-8")
    return {
        "title": page["title"],
        "url": page_url,
        "last_updated": page["version"]["createdAt"],
        "assistant": "shared",
    }


def check_response(response: httpx.Response) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"Confluence returned HTTP {response.status_code}; "
            "check the cloud ID, approved space, token expiry, and "
            "read:space:confluence / read:page:confluence / "
            "read:content.metadata:confluence scopes."
        )


def main(
    *,
    before_ingest: Callable[[], None] | None = None,
    image_report_sink: Callable[[dict], None] | None = None,
) -> None:
    # HTTP request URLs can contain signed media tokens; never put them in pipeline logs.
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    names = (
        "CONFLUENCE_BASE_URL",
        "CONFLUENCE_CLOUD_ID",
        "CONFLUENCE_SPACE_KEY",
        "CONFLUENCE_ROOT_PAGE_ID",
        "CONFLUENCE_EMAIL",
        "CONFLUENCE_API_TOKEN",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing Confluence import configuration: {', '.join(missing)}")
    base_url = validate_base_url(os.environ["CONFLUENCE_BASE_URL"])
    space = os.environ["CONFLUENCE_SPACE_KEY"]
    root_page_id = os.environ["CONFLUENCE_ROOT_PAGE_ID"]
    vision = os.environ.get("CONFLUENCE_IMPORT_IMAGES", "false").lower()
    if vision not in {"true", "false"}:
        raise ValueError("CONFLUENCE_IMPORT_IMAGES must be true or false")
    with (
        tempfile.TemporaryDirectory(prefix="cloudops-confluence-") as directory,
        ExitStack() as stack,
    ):
        source = Path(directory) / "pages"
        images = None
        completed = False
        if vision == "true":
            settings = get_settings()
            if not settings.azure_openai_endpoint or not settings.azure_openai_deployment:
                raise RuntimeError("Confluence vision requires OpenAI endpoint and deployment")
            credential = DefaultAzureCredential()
            stack.callback(credential.close)
            ai = stack.enter_context(create_openai_client(settings, credential))
            media_client = stack.enter_context(httpx.Client(timeout=30, follow_redirects=False))
        with httpx.Client(
            auth=(os.environ["CONFLUENCE_EMAIL"], os.environ["CONFLUENCE_API_TOKEN"]),
            timeout=30,
            follow_redirects=False,
        ) as client:
            if vision == "true":
                images = ConfluenceImages(
                    client,
                    media_client,
                    ai,
                    settings,
                    base_url,
                    str(UUID(os.environ["CONFLUENCE_CLOUD_ID"])),
                )
                if image_report_sink:

                    def publish_image_report():
                        image_report_sink(
                            {
                                "status": "imported" if completed else "failed",
                                "finished_at": datetime.now(UTC).isoformat(),
                                "images_imported": images.images,
                                "images_skipped": images.skipped,
                                "failures": images.failures,
                            }
                        )

                    stack.callback(publish_image_report)
            metadata = export_space(
                client,
                base_url,
                space,
                source,
                cloud_id=os.environ["CONFLUENCE_CLOUD_ID"],
                root_page_id=root_page_id,
                images=images,
            )
        if before_ingest:
            before_ingest()
        ingest(
            source,
            source_id=f"confluence:{urlparse(base_url).hostname}:{space}:page:{root_page_id}",
            version=datetime.now(UTC).isoformat(),
            source_kind="operations",
            metadata=metadata,
            before_write=before_ingest,
        )
        completed = True
    print(
        json.dumps(
            {
                "status": "imported",
                "pages": len(metadata),
                "space": space,
                "images": images.images if images else 0,
                "unsupported_image_references": images.skipped if images else 0,
                "image_failures": images.failures if images else [],
            }
        )
    )


if __name__ == "__main__":
    main()
