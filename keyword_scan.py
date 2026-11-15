"""Scheduled glossary keyword scan. Runs only as its own Azure Container Apps Job, never in chat.

Each run reads logged questions newer than the saved checkpoint in the scan-state blob,
adds missing keywords to Admin > Glossary review, and advances the checkpoint per page
only after that page is saved. Old logs are therefore never read again.
"""

import logging
import sys

from cloudops_hub.config import Settings, get_settings
from cloudops_hub.glossary_review import scan_logged_keywords
from cloudops_hub.question_log import QuestionLog

logger = logging.getLogger(__name__)
# 5 questions per page; bounds one execution to 10,000 questions. The next run resumes.
MAX_PAGES = 2_000


def run(settings: Settings, questions) -> dict:
    scanned = 0
    warning = ""
    for _ in range(MAX_PAGES):
        page = scan_logged_keywords(settings, questions)
        scanned += page["scanned"]
        warning = page.get("warning", "")
        if not page["next_cursor"]:
            break
    else:
        logger.warning("Keyword scan reached its page limit; the next run continues")
    return {"scanned": scanned, "warning": warning}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if not settings.azure_storage_account_url:
        logger.error("Keyword scan requires AZURE_STORAGE_ACCOUNT_URL")
        return 1
    try:
        result = run(settings, QuestionLog(settings))
    except Exception as exc:  # noqa: BLE001 - fail the execution; saved pages stay committed
        logger.error("Keyword scan failed (%s); the next run resumes", type(exc).__name__)
        return 1
    logger.info("Keyword scan processed %d new logged questions", result["scanned"])
    if result["warning"]:
        logger.warning("Keyword scan: %s", result["warning"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
