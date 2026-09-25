"""
AssureX Claim Engine - Claim Summary Card Generator (v2 - IMPROVED)
--------------------------------------------------------------------
WHY v2? Testing proved the v1 card design (plain text only) was too
hard for the Teachable Machine image model to learn from -- after
shrinking to 224x224, small text differences ("Yes" vs "No", "Active"
vs "Expired") become nearly unreadable, and the model couldn't
reliably tell Invalid claims apart from ManualReview claims (only 17%
correct on real held-out validation images).

v2 adds a prominent RISK INDICATOR GRID near the top of the card: a
row of big colored squares (green = OK, red = flag) representing the
same underlying facts, just in a format that survives image shrinking
much better than small text does. Convolutional image models are
naturally good at recognizing color/shape patterns -- this plays to
that strength instead of asking it to read fine text.

IMPORTANT: this does NOT leak the model's prediction or add any new
information -- it's the exact same facts (warranty expired? excluded
damage? missing docs? etc), just drawn as color blocks instead of
words. The detailed text list is still included below the grid for
human readability and completeness.

This file provides the same function signature as before:
    generate_card(claim, variation) -> PIL Image
so nothing else in the pipeline (card generation script, Flask route)
needs to change.
"""

import os
import platform
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))

GREEN = "#2E7D32"
RED = "#C62828"

STYLES = [
    {  # variation 0
        "bg_color": "#FFFFFF",
        "header_bg": "#2B6CB0",
        "header_text": "#FFFFFF",
        "label_color": "#333333",
        "value_color": "#000000",
        "divider_color": "#DDDDDD",
        "title_font_size": 38,
        "body_font_size": 28,
        "margin": 40,
        "line_gap": 34,
    },
    {  # variation 1
        "bg_color": "#F7F7F2",
        "header_bg": "#37474F",
        "header_text": "#FFFFFF",
        "label_color": "#555555",
        "value_color": "#111111",
        "divider_color": "#CCCCCC",
        "title_font_size": 36,
        "body_font_size": 28,
        "margin": 50,
        "line_gap": 34,
    },
]

CARD_WIDTH = 700
CARD_HEIGHT = 780
# IMPORTANT: this height is deliberately kept close to CARD_WIDTH.
# Teachable Machine (and our own code) crops the taller dimension to make
# a square before resizing -- it removes pixels EQUALLY from the top and
# bottom. A tall card (we originally used 980) meant the crop removed 140px
# from the top, which completely destroyed the risk-indicator grid sitting
# right below the header. Keeping height close to width means the crop only
# removes ~40px from each side -- enough to trim the header a little
# (harmless, it carries no class information) without touching the grid or
# any field row. This was verified visually: see what_the_model_actually_sees.png.

DISPLAY_FIELDS = [
    ("claim_id", "Claim ID"),
    ("product_category", "Product Category"),
    ("product_age_months", "Product Age (months)"),
    ("warranty_duration_months", "Warranty Duration (months)"),
    ("warranty_status", "Warranty Status"),
    ("fault_type", "Fault Type"),
    ("repair_count", "Previous Repairs"),
    ("authorized_repair", "Authorized Repair"),
    ("has_receipt", "Receipt Uploaded"),
    ("has_warranty_card", "Warranty Card Uploaded"),
    ("has_product_image", "Product Image Uploaded"),
    ("has_serial_evidence", "Serial Evidence Uploaded"),
    ("missing_doc_count", "Missing Documents"),
    ("serial_match", "Serial Number Match"),
    ("excluded_damage", "Excluded Damage Type"),
    ("duplicate_claim", "Duplicate Claim Suspected"),
]

# Each risk indicator: (short label for under the square, function that
# returns True if this is a RED FLAG for the given claim)
RISK_INDICATORS = [
    ("Warranty", lambda c: c.get("warranty_status") == "Expired"),
    ("Excluded", lambda c: c.get("excluded_damage") == "Yes"),
    ("Serial", lambda c: c.get("serial_match") == "No"),
    ("Duplicate", lambda c: c.get("duplicate_claim") == "Yes"),
    ("Repair Auth", lambda c: c.get("authorized_repair") == "No"),
    ("Receipt", lambda c: c.get("has_receipt") == "No"),
    ("W.Card", lambda c: c.get("has_warranty_card") == "No"),
    ("Photo", lambda c: c.get("has_product_image") == "No"),
    ("S/N Evid.", lambda c: c.get("has_serial_evidence") == "No"),
]


def _load_font(size, bold=False):
    """
    Cross-platform font loader.
    Priority:
      1. Bundled fonts in card_generator/fonts/ (portable, works everywhere)
      2. Windows system fonts: Arial / Arial Bold (always present on Windows)
      3. Linux system fonts: DejaVu (for deployment servers / Render / Railway)
      4. PIL default bitmap font (last resort, tiny but never crashes)
    """
    deja_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    win_name  = "arialbd.ttf"         if bold else "arial.ttf"

    candidates = [
        # 1. Bundled fonts next to this script
        os.path.join(_HERE, "fonts", deja_name),
        # 2. Windows system fonts (always available on any Windows machine)
        os.path.join("C:/Windows/Fonts", win_name),
        os.path.join("C:/Windows/Fonts", deja_name),
        # 3. Linux / macOS system fonts
        f"/usr/share/fonts/truetype/dejavu/{deja_name}",
        f"/usr/share/fonts/truetype/msttcorefonts/{win_name}",
        f"/Library/Fonts/{win_name}",
    ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Last resort — PIL built-in bitmap font (tiny, but never crashes)
    return ImageFont.load_default()


def _draw_risk_grid(draw, claim, top_y, margin, style):
    """
    Draws a row of big colored squares -- one per risk indicator -- with
    a small label under each. Green = fine, Red = flag. This is the
    part specifically designed to survive being shrunk to 224x224.
    """
    n = len(RISK_INDICATORS)
    available_width = CARD_WIDTH - 2 * margin
    square_size = 56
    gap = (available_width - n * square_size) / (n - 1)

    label_font = _load_font(12)
    x = margin

    for label, is_flag_fn in RISK_INDICATORS:
        is_flag = is_flag_fn(claim)
        color = RED if is_flag else GREEN

        draw.rectangle([x, top_y, x + square_size, top_y + square_size], fill=color)

        # small label centered under the square, wrapped if needed
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (square_size - text_w) / 2
        draw.text((text_x, top_y + square_size + 6), label, font=label_font, fill="#333333")

        x += square_size + gap

    return top_y + square_size + 30  # returns the y-position to continue drawing below


def generate_card(claim: dict, variation: int = 0) -> Image.Image:
    style = STYLES[variation % len(STYLES)]

    img = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), style["bg_color"])
    draw = ImageDraw.Draw(img)

    title_font = _load_font(style["title_font_size"], bold=True)
    label_font = _load_font(style["body_font_size"])
    value_font = _load_font(style["body_font_size"], bold=True)

    # ---- Header bar ----
    header_height = 70
    draw.rectangle([0, 0, CARD_WIDTH, header_height], fill=style["header_bg"])
    draw.text((style["margin"], 18), "Claim Summary Card",
              font=title_font, fill=style["header_text"])

    # ---- Risk indicator grid (the new part) ----
    grid_top_y = header_height + 24
    next_y = _draw_risk_grid(draw, claim, grid_top_y, style["margin"], style)

    draw.line([(style["margin"], next_y), (CARD_WIDTH - style["margin"], next_y)],
               fill=style["divider_color"], width=2)
    next_y += 20

    # ---- Detailed text fields (same as v1, for completeness/readability) ----
    y = next_y
    for field_key, field_label in DISPLAY_FIELDS:
        value = claim.get(field_key, "N/A")

        draw.text((style["margin"], y), f"{field_label}:",
                  font=label_font, fill=style["label_color"])
        draw.text((style["margin"] + 300, y), str(value),
                  font=value_font, fill=style["value_color"])

        y += style["line_gap"]
        draw.line([(style["margin"], y - 8), (CARD_WIDTH - style["margin"], y - 8)],
                   fill=style["divider_color"], width=1)

    return img


# ---------------------------------------------------------------------
# Quick self-test: generates a clean claim AND a risky claim side by
# side so you can visually compare the risk grids.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    clean_claim = {
        "claim_id": "CLM00001", "product_category": "Washing Machine",
        "product_age_months": 5, "warranty_duration_months": 12,
        "warranty_status": "Active", "fault_type": "Mechanical Failure",
        "repair_count": 0, "authorized_repair": "Yes", "has_receipt": "Yes",
        "has_warranty_card": "Yes", "has_product_image": "Yes",
        "has_serial_evidence": "Yes", "missing_doc_count": 0,
        "serial_match": "Yes", "excluded_damage": "No", "duplicate_claim": "No",
    }

    risky_claim = {
        "claim_id": "CLM00004", "product_category": "Laptop",
        "product_age_months": 40, "warranty_duration_months": 12,
        "warranty_status": "Expired", "fault_type": "Physical Damage",
        "repair_count": 3, "authorized_repair": "No", "has_receipt": "No",
        "has_warranty_card": "No", "has_product_image": "No",
        "has_serial_evidence": "No", "missing_doc_count": 4,
        "serial_match": "No", "excluded_damage": "Yes", "duplicate_claim": "Yes",
    }

    generate_card(clean_claim, variation=0).save("sample_clean_v0.png")
    generate_card(risky_claim, variation=0).save("sample_risky_v0.png")
    print("Saved sample_clean_v0.png and sample_risky_v0.png")
