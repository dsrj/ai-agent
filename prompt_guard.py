"""Deny-list filter that stops known prompt-injection attempts before any model call.

This is a first line of defence only; the model-side guardrails still apply to anything that
passes. Patterns are deliberately narrow so normal CloudOps wording (Terraform
``ignore_changes``, firewall "rules", CLI "prompts") is not blocked.
"""

from __future__ import annotations

import re
import unicodedata

BLOCKED_ANSWER = (
    "This message looks like an attempt to change the assistant's instructions, so it was "
    "not sent to the AI. Ask your CloudOps question directly."
)

_INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")

_VERB = r"(?:ignore|disregard|forget|override|overwrite)"
_SCOPE = (
    r"(?:\s+(?:all|any|of|the|your|my|these|those|previous|prior|above|earlier|preceding"
    r"|initial|original|system|developer|safety|existing|current|other))*"
)
_STRONG_SCOPE = r"(?:your|previous|prior|above|earlier|preceding|initial|original|system)"

_DA_VERB = r"(?:ignorer|glem|se\s+bort\s+fra|tilsidesæt|overskriv)"
_DA_SCOPE = r"(?:\s+(?:alle|de|dine|din|tidligere|ovenstående|forrige|oprindelige|system))*"

PATTERNS = [
    # English: override earlier instructions
    rf"\b{_VERB}{_SCOPE}\s+(?:instructions?|guidelines?|directions?|directives?)\b",
    rf"\b{_VERB}{_SCOPE}\s+{_STRONG_SCOPE}\s+(?:prompts?|rules|messages?|context)\b",
    r"\b(?:bypass|circumvent|disable|turn\s+off|get\s+around|break)\s+(?:all\s+)?"
    r"(?:your|the\s+ai'?s?|safety|content)\s+(?:safety\s+|content\s+)?"
    r"(?:rules|restrictions|filters?|guardrails?|policies|safeguards|instructions)\b",
    r"\b(?:bypass|circumvent|disable|turn\s+off|get\s+around)\s+(?:all\s+|the\s+|any\s+)?"
    r"(?:guardrails?|safeguards|content\s+filters?|safety\s+filters?)\b",
    # English: prompt exfiltration
    r"\b(?:reveal|show|print|repeat|output|display|tell\s+me|give\s+me|leak|dump|what\s+(?:is|are))"
    r"\b[^.?!\n]{0,30}\b(?:your|the)\s+(?:system|hidden|initial|original|secret)\s+"
    r"(?:prompt|instructions?|message|rules)",
    r"\b(?:repeat|print|output)\b[^.?!\n]{0,20}\b(?:everything|all|the\s+text|the\s+words)\s+above\b",
    # English: role-play / jailbreak
    r"\byou\s+are\s+now\s+(?:a|an|in|dan|free|unrestricted|unfiltered|uncensored|no\s+longer)\b",
    r"\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should|have|can)\b",
    r"\bpretend\s+(?:to\s+be|you\s+are|you're|that\s+you)\b",
    r"\bact\s+as\s+(?:dan\b|an?\s+(?:unrestricted|unfiltered|uncensored|evil|different|new)\b"
    r"|if\s+you\s+(?:have|had)\s+no\b)",
    r"\b(?:roleplay|role-play|role\s+play)\s+as\b",
    r"\b(?:you\s+are|you're)\s+(?:now\s+)?in\s+developer\s+mode\b",
    r"\b(?:enter|activate)\s+(?:developer|god|dan)\s+mode\b",
    r"\bjail\s*break",
    r"\bdo\s+anything\s+now\b",
    r"\bdan\s+mode\b",
    r"\b(?:answer|respond|reply|talk|speak|act|behave|operate)\b[^.?!\n]{0,20}"
    r"\b(?:without|with\s+no)\s+(?:any\s+)?(?:restrictions|filters|limitations|guardrails|censorship)\b",
    r"\bnew\s+(?:system\s+)?instructions?\s*:",
    # Chat-template / role-injection tokens
    r"<\|\s*(?:im_start|im_end|system|endoftext|user|assistant)\s*\|>",
    r"\[\s*/?\s*(?:system|inst)\s*\]",
    r"(?:^|\n)\s*#{2,}\s*(?:system|instruction)s?\b",
    # Danish
    rf"\b{_DA_VERB}{_DA_SCOPE}\s+(?:instruktion(?:er|erne)?|instruks(?:er|erne)?"
    r"|systemprompt(?:en)?|retningslinjer(?:ne)?)\b",
    rf"\b{_DA_VERB}{_DA_SCOPE}\s+(?:dine|tidligere|ovenstående|oprindelige)\s+regler\b",
    r"\b(?:vis|gentag|udskriv|fortæl)\b[^.?!\n]{0,30}\b(?:din|dine|den)\s+"
    r"(?:system\s*prompt|systemprompt|skjulte\s+instruktioner|oprindelige\s+instruktioner)",
    r"\bomgå\s+(?:dine|alle)\s+(?:regler|begrænsninger|filtre|sikkerhedsregler)\b",
    r"\bdu\s+er\s+nu\s+(?:en|et|i|dan|fri|ubegrænset|ufiltreret)\b",
    r"\bfra\s+nu\s+af\s+(?:er|skal|vil|må|kan)\s+du\b",
    r"\blad\s+som\s+om\s+(?:du|at\s+du)\b",
    r"\b(?:opfør\s+dig|agér|ager|spil)\s+som\s+(?:dan\b|en\s+(?:ubegrænset|ufiltreret|anden))",
    r"\b(?:svar|svarer|besvar|tal|skriv|opfør\s+dig)\b[^.?!\n]{0,20}\buden\s+(?:nogen\s+)?"
    r"(?:begrænsninger|filtre|restriktioner|regler|censur)\b",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in PATTERNS]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE.sub("", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.casefold()


def is_prompt_attack(text: str) -> bool:
    normalized = normalize(text)
    return any(pattern.search(normalized) for pattern in _COMPILED)
