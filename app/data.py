"""Synthetic data for all 5 tasks."""
from app.models import Invoice, PurchaseOrder, LineItem


# TASKS 1-3 (Existing)
TASK1_INVOICE = Invoice(
    id="INV-2024-0041",
    raw_text=(
        "1NVO1CE\n"
        "Vend0r: Apex Office Supplies Ltd.\n"
        "1nvoice No: INV-2024-OO41\n"
        "Date: 2024-03-15\n"
        "L1NE ITEMS:\n"
        "1. Ergonomic Office Chair (x4) @ $189.99 = $759.96\n"
        "2. Standing Desk Converter (x2) @ $349.50 = $699.OO\n"
        "3. Wireless Keyboard (x6) @ $54.75 = $328.50\n"
        "Subtota1: $1,787.46\n"
        "Tax: $151.93\n"
        "Total Due: $1,939.39"
    ),
)

TASK1_GROUND_TRUTH = {
    "vendor_name": "Apex Office Supplies Ltd.",
    "invoice_number": "INV-2024-0041",
    "invoice_date": "2024-03-15",
    "line_items": [{}, {}, {}],
    "subtotal": 1787.46,
    "tax_amount": 151.93,
    "total_amount": 1939.39,
}

TASK2_PO = PurchaseOrder(
    po_number="PO-2024-0887",
    vendor_name="BlueSky",
    line_items=[
        LineItem(description="Cardboard Shipping Box (Pack of 50)", quantity=10, unit_price=60.00, total=600.00),
        LineItem(description="Bubble Wrap", quantity=50, unit_price=12.00, total=600.00),
    ],
    total_amount=1900.00,
    issued_date="2024-02-01",
    valid_until="2024-04-30",
)

TASK2_INVOICE = Invoice(
    id="INV-BSL-0112",
    raw_text=(
        "Cardboard (Individual) x 600 @ $1.20 = $720.00\n"
        "Bubble Wrap x 50 @ $14.00 = $700.00\n"
        "Total: $2120.00"
    ),
)

TASK2_GROUND_TRUTH = {
    "decision": "reject",
    "mismatches": ["quantity", "price", "total"],
}

TASK3_VENDOR_WHITELIST = ["Apex", "BlueSky"]
TASK3_INVOICES = [
    Invoice(id="INV-1", raw_text="Valid"),
    Invoice(id="INV-2", raw_text="Amount Mismatch"),
    Invoice(id="INV-3", raw_text="URGENT FINAL NOTICE"),
    Invoice(id="INV-4", raw_text="Unauthorized Vendor"),
    Invoice(id="INV-5", raw_text="Duplicate"),
]

TASK3_GROUND_TRUTH = {
    "fraudulent_ids": ["INV-2", "INV-4", "INV-5"],
    "fraud_reasons": {
        "INV-2": "amount",
        "INV-4": "vendor",
        "INV-5": "duplicate",
    },
}

# -- TASK 4: GL Coding ---------------------------------------------------------
TASK4_CHART_OF_ACCOUNTS = {
    "GL-5000": "Office Meals & Entertainment",
    "GL-6000": "IT Hardware & Equipment",
    "GL-6100": "Software & Cloud Services",
    "GL-7000": "General Office Supplies",
}

TASK4_INVOICE = Invoice(
    id="INV-GL-992",
    raw_text="""
ITEMS BILLED:
1. "Dell XPS 15 Laptop (Engineering)" - $2,500.00
2. "GitHub Enterprise License (Annual)" - $1,200.00
3. "Team Pizza Lunch (Q3 Kickoff)" - $350.00
4. "Printer Ink Cartridges (Cyan/Magenta)" - $120.00
""",
)

TASK4_GROUND_TRUTH = {
    "Dell XPS 15 Laptop (Engineering)": "GL-6000",
    "GitHub Enterprise License (Annual)": "GL-6100",
    "Team Pizza Lunch (Q3 Kickoff)": "GL-5000",
    "Printer Ink Cartridges (Cyan/Magenta)": "GL-7000",
}

# -- TASK 5: Statement Reconciliation -----------------------------------------
TASK5_STATEMENT = """
VENDOR STATEMENT - TECH_CORP_LLC
Date: 2024-04-01
-------------------------------------------------
DATE       | INV NUMBER | AMOUNT  | STATUS
2024-03-01 | INV-8001   | $500.00 | OPEN
2024-03-05 | INV-8002   | $250.00 | OPEN
2024-03-10 | INV-8003   | $100.00 | OPEN
2024-03-15 | INV-8004   | $950.00 | OPEN
"""

TASK5_LEDGER = [
    {"invoice_number": "INV-8001", "amount_logged": 500.00, "status": "Paid"},
    {"invoice_number": "INV-8002", "amount_logged": 200.00, "status": "Pending Verification"},
    {"invoice_number": "INV-8004", "amount_logged": 950.00, "status": "Approved for Payment"},
]

TASK5_GROUND_TRUTH = {
    "missing_invoices": ["INV-8003"],
    "discrepancy_invoices": ["INV-8002"],
}
