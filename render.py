"""Render the scanner card grid to a self-contained HTML file.

The static export and the live /scanner page share one template: the live page
leaves the embedded block empty and fetches /api/scan, while the export bakes
the scan result straight into the page so the file works with no server.
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).with_name("templates")


def render_static(result: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("scanner.html")
    # </script> inside the payload would close the host block early
    payload = json.dumps(result).replace("</", "<\\/")
    return template.render(embedded=payload)
