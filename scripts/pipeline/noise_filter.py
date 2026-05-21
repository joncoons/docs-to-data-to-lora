"""Stage 0 noise filter — drop passages that aren't worth generating Q+A from."""
import re

import tiktoken

_LICENSE_PREFIXES = (
    "apache license", "mit license", "copyright (c)", "bsd license",
    "gnu general public", "mozilla public license", "terms and conditions",
)

_BLOCKED_URL_SEGMENTS = (
    "acknowledgement", "acknowledge", "third-party", "thirdparty",
    "open-source", "opensource", "legal", "license-notice",
)

_LICENSE_BOILERPLATE_RE = re.compile(
    r"(apache\.org/licenses/LICENSE-2\.0"
    r"|WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND"
    r"|THIS SOFTWARE IS PROVIDED\s+[\"']?AS[- ]IS"
    r"|Redistribution and use in source and binary forms"
    r"|Permission is hereby granted, free of charge"
    r"|under the terms of the GNU"
    r"|Lesser General Public License"
    r"|Mozilla Public License"
    r"|creativecommons\.org/licenses)",
    re.IGNORECASE | re.DOTALL,
)

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def is_noise(text: str, url: str = "", min_tokens: int = 60) -> bool:
    stripped = text.strip()
    lower = stripped.lower()

    if count_tokens(stripped) < min_tokens:
        return True

    if url:
        url_lower = url.lower()
        if any(seg in url_lower for seg in _BLOCKED_URL_SEGMENTS):
            return True

    if any(lower.startswith(p) for p in _LICENSE_PREFIXES):
        return True

    if _LICENSE_BOILERPLATE_RE.search(stripped):
        return True

    lines = stripped.splitlines()
    if len(lines) > 5:
        nav_lines = sum(1 for l in lines if re.match(r"^- \[", l.strip()))
        if nav_lines / len(lines) > 0.60:
            return True

    code_chars = sum(len(m.group(0)) for m in re.finditer(r"```.*?```", stripped, re.DOTALL))
    if len(stripped) > 0 and code_chars / len(stripped) > 0.85:
        return True

    return False
