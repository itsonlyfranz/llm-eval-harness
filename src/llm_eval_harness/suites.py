"""Synthetic test suites. Every input is invented or drawn from the target repos' bundled sample data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    target: str
    case_id: str
    task: str  # what the agent is asked, shown to the judge
    rubric: str
    reference: str  # ground truth shown to the judge
    expect: dict[str, Any] = field(default_factory=dict)  # drives the deterministic checks


STRUCTURED = [
    Case("structured-output", "so-01-plain", "Extract the invoice: 'Invoice from Acme Corp. Amount due: 120.50 USD.'",
         "vendor, total and currency must match the text exactly", '{"vendor": "Acme Corp", "total": 120.5, "currency": "USD"}',
         {"text": "Invoice from Acme Corp. Amount due: 120.50 USD.", "vendor": "Acme Corp", "total": 120.5, "currency": "USD"}),
    Case("structured-output", "so-02-euro-symbol", "Extract the invoice: 'Globex GmbH billed us €2,340.00 for March consulting.'",
         "currency must be the ISO code EUR, total 2340", '{"vendor": "Globex GmbH", "total": 2340.0, "currency": "EUR"}',
         {"text": "Globex GmbH billed us €2,340.00 for March consulting.", "vendor": "Globex GmbH", "total": 2340.0, "currency": "EUR"}),
    Case("structured-output", "so-03-subtotal-vs-total", "Extract the invoice: 'Initech Ltd. Subtotal 800.00 GBP, VAT 160.00 GBP, total payable 960.00 GBP.'",
         "total must be the amount payable, not the subtotal", '{"vendor": "Initech Ltd", "total": 960.0, "currency": "GBP"}',
         {"text": "Initech Ltd. Subtotal 800.00 GBP, VAT 160.00 GBP, total payable 960.00 GBP.", "vendor": "Initech Ltd", "total": 960.0, "currency": "GBP"}),
    Case("structured-output", "so-04-thousands", "Extract the invoice: 'Umbrella Logistics invoice no. 7781: 1,250,000 JPY due in 30 days.'",
         "total must be 1250000 (the invoice number is not the amount)", '{"vendor": "Umbrella Logistics", "total": 1250000.0, "currency": "JPY"}',
         {"text": "Umbrella Logistics invoice no. 7781: 1,250,000 JPY due in 30 days.", "vendor": "Umbrella Logistics", "total": 1250000.0, "currency": "JPY"}),
    Case("structured-output", "so-05-dollar-cad", "Extract the invoice: 'Maple Leaf Supplies sent a bill for 89.99 Canadian dollars (CAD).'",
         "currency CAD, total 89.99", '{"vendor": "Maple Leaf Supplies", "total": 89.99, "currency": "CAD"}',
         {"text": "Maple Leaf Supplies sent a bill for 89.99 Canadian dollars (CAD).", "vendor": "Maple Leaf Supplies", "total": 89.99, "currency": "CAD"}),
    Case("structured-output", "so-06-discount", "Extract the invoice: 'Stark Components: list price 500 USD, 20% discount applied, final charge 400 USD.'",
         "total must be the final charge after discount", '{"vendor": "Stark Components", "total": 400.0, "currency": "USD"}',
         {"text": "Stark Components: list price 500 USD, 20% discount applied, final charge 400 USD.", "vendor": "Stark Components", "total": 400.0, "currency": "USD"}),
    Case("structured-output", "so-07-lowercase-vendor", "Extract the invoice: 'from: pixel & co. -- total 42.00 chf'",
         "vendor pixel & co (any casing), currency CHF, total 42", '{"vendor": "pixel & co.", "total": 42.0, "currency": "CHF"}',
         {"text": "from: pixel & co. -- total 42.00 chf", "vendor": "pixel & co", "total": 42.0, "currency": "CHF"}),
    Case("structured-output", "so-08-two-amounts", "Extract the invoice: 'Wayne Freight billed 15.00 USD for insurance and 310.00 USD for shipping; grand total 325.00 USD.'",
         "total must be the grand total 325", '{"vendor": "Wayne Freight", "total": 325.0, "currency": "USD"}',
         {"text": "Wayne Freight billed 15.00 USD for insurance and 310.00 USD for shipping; grand total 325.00 USD.", "vendor": "Wayne Freight", "total": 325.0, "currency": "USD"}),
]

RAG_RUBRIC = "The answer must state the facts in the reference and add nothing the reference does not support. An abstention is correct only when the reference says the question cannot be answered."


def _rag(cid: str, q: str, reference: str, must: list[str], answerable: bool = True, fallback: bool = False) -> Case:
    return Case("rag-citation", cid, q, RAG_RUBRIC, reference,
                {"question": q, "must_contain": must, "answerable": answerable, "fallback": fallback})


RAG = [
    _rag("rag-01-audit-ent", "How long are audit logs retained on the Enterprise plan?", "400 days.", ["400"]),
    _rag("rag-02-team-price", "How much does the Team plan cost per month?", "$79 per month.", ["79"]),
    _rag("rag-03-refund", "How long do new customers have to request a full refund?", "14 days from the first payment.", ["14"]),
    _rag("rag-04-sso", "Which plan supports SAML single sign-on?", "Only the Enterprise plan.", ["enterprise"]),
    _rag("rag-05-credit", "What service credit applies if monthly uptime falls below 99.0%?", "A credit of 25% of the monthly fee.", ["25"]),
    _rag("rag-06-ent-response", "How fast does Enterprise support respond to critical incidents?", "Within 1 hour.", ["1 hour"]),
    _rag("rag-07-backups", "How long are backups kept after the original data is deleted?", "35 days.", ["35"]),
    _rag("rag-08-regions", "Which regions does Helios Cloud operate data centers in?", "Five regions: US East (Virginia), US West (Oregon), EU Central (Frankfurt), EU West (Dublin), Asia Pacific (Singapore). Comes from the web snapshot, not the KB.", ["singapore"], fallback=True),
    _rag("rag-09-ceo", "Who is the CEO of Helios Cloud?", "Not answerable from the documents; the agent should abstain.", [], answerable=False),
    _rag("rag-10-worldcup", "Who won the 2022 football World Cup?", "Not answerable from the documents; the agent should abstain.", [], answerable=False),
]

REACT_RUBRIC = "The final answer must give the correct value from the reference. If the reference says the task is unanswerable, the agent must say it cannot answer and must not guess."


def _react(cid: str, task: str, reference: str, must: str | None) -> Case:
    return Case("react-planning", cid, task, REACT_RUBRIC, reference, {"task": task, "must_contain": must})


REACT = [
    _react("re-01-arith", "What is 17 * 23 + 4?", "395", "395"),
    _react("re-02-arith-paren", "Compute (120 - 45) / 5.", "15", "15"),
    _react("re-03-lookup", "How tall is the Eiffel Tower in metres?", "330 metres", "330"),
    _react("re-04-lookup-add", "What is the combined population of Paris and Berlin, in millions?", "2.1 + 3.7 = 5.8 million", "5.8"),
    _react("re-05-difference", "How many more million residents does Berlin have than Paris?", "3.7 - 2.1 = 1.6 million", "1.6"),
    _react("re-06-ratio", "Divide Tokyo's population by Paris's population and round to one decimal place.", "9.7 / 2.1 = 4.62, rounds to 4.6", "4.6"),
    _react("re-07-pct", "Take the height of the Eiffel Tower in metres and add 10 percent. What is the result?", "330 * 1.1 = 363", "363"),
    _react("re-08-unanswerable", "What is the population of Rome?", "Unanswerable: the fact base has no entry for Rome. The agent must say it cannot find it and must not invent a number.", None),
]

SUITES = {"structured-output": STRUCTURED, "rag-citation": RAG, "react-planning": REACT}
