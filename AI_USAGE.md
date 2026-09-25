# AI Tool Usage Declaration
# AssureX Claim Engine — Competition Project

This file declares all AI tools used during the development of the AssureX Claim Engine,
as required by Section 1.8 item 9 of the SRS.

---

## Tool 1: Antigravity IDE (Google DeepMind)

| Field | Detail |
|-------|--------|
| **Tool Name** | Antigravity IDE (AI coding assistant) |
| **Purpose of Use** | Code review, bug identification, cross-platform compatibility fixes |
| **Modules Affected** | `card_generator.py`, `compare_predictions.py`, `app.py` |
| **Prompts / Type of Assistance** | Identifying hardcoded Linux font path; fixing label-value column collision in card layout; reviewing SRS compliance gaps |
| **Modifications by Team** | Team reviewed all suggested changes, tested generated card images visually, verified font rendering on Windows, confirmed no prediction data leaked into cards |
| **Testing Performed** | Ran `python card_generator.py` self-test, visually inspected generated PNG cards, confirmed exit code 0 |
| **Verified by** | Maheen |

---

## Tool 2: GitHub Copilot / ChatGPT (if used)

| Field | Detail |
|-------|--------|
| **Tool Name** | N/A — declare here if used |
| **Purpose of Use** | N/A |
| **Modules Affected** | N/A |
| **Prompts / Type of Assistance** | N/A |
| **Modifications by Team** | N/A |
| **Testing Performed** | N/A |
| **Verified by** | N/A |

---

## Important Notice

> AI-generated output was NOT used to replace team understanding.
> All code was reviewed, modified where necessary, and independently tested by team members.
> The final claim decision is produced entirely by the team's Python classification model,
> Google Teachable Machine model, warranty rule engine, and application logic — NOT through
> any external generative AI API.
