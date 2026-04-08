"""
Synthetic invoice and purchase-order data for all three tasks.
All data is deterministic. No randomness. No file I/O.
"""

from app.models import Invoice, PurchaseOrder, LineItem


# -- TASK 1: OCR Noise Injection ----------------------------------------------

TASK1_INVOICE = Invoice(
    id="INV-2024-0041",
    raw_text="""
1NVO1CE  [SCAN_REF: 88291]

Vend0r:        Apex Office Supplies Ltd.
1nvoice No:    INV-2024-OO41
Date:          2024-03-15
Bi11 To:       Meridian Corp, 400 Market Street, San Francisco, CA 94105

L1NE ITEMS:
    1. Ergonomic Office Chair (x4)        @ $189.99 ea      =  $759.96
    2. Standing Desk Converter (x2)       @ $349.50 ea      =  $699.OO
    3. Wireless Keyboard & Mouse Set (x6) @ $54.75 ea       =  $328.50

Subtota1:   $1,787.46
Tax(8.5%):    $151.93
Total Due:  $1,939.39

Pymt Terms: Net 30
""",
        metadata={"source": "poor_quality_scan", "confidence": 0.72},
)

# Ground truth remains perfect. The agent must fix the OCR errors.
TASK1_GROUND_TRUTH = {
    "vendor_name": "Apex Office Supplies Ltd.",
    "invoice_number": "INV-2024-0041",
    "invoice_date": "2024-03-15",
    "line_items": [
        {"description": "Ergonomic Office Chair", "quantity": 4, "unit_price": 189.99, "total": 759.96},
        {"description": "Standing Desk Converter", "quantity": 2, "unit_price": 349.50, "total": 699.00},
        {"description": "Wireless Keyboard & Mouse Set", "quantity": 6, "unit_price": 54.75, "total": 328.50},
    ],
    "subtotal": 1787.46,
    "tax_amount": 151.93,
    "total_amount": 1939.39,
}


# -- TASK 2: Unit of Measure Math ---------------------------------------------

TASK2_PO = PurchaseOrder(
    po_number="PO-2024-0887",
    vendor_name="BlueSky Logistics Inc.",
    line_items=[
        LineItem(description="Industrial Packing Tape (48mm x 100m)", quantity=200, unit_price=3.50, total=700.00),
        # PO specifies 10 packs of 50 = 500 total boxes
        LineItem(description="Cardboard Shipping Box (Pack of 50)", quantity=10, unit_price=60.00, total=600.00),
        LineItem(description="Bubble Wrap Roll (50m)", quantity=50, unit_price=12.00, total=600.00),
    ],
    total_amount=1900.00,
    currency="USD",
    issued_date="2024-02-01",
    valid_until="2024-04-30",
)

TASK2_INVOICE = Invoice(
    id="INV-BSL-2024-0112",
    raw_text="""
INVOICE

From:          BlueSky Logistics Inc.
Invoice #:     INV-BSL-2024-0112
Date:          2024-03-20
PO Reference:  PO-2024-0887
Bill To:       Meridian Corp

LINE ITEMS:
    1. Industrial Packing Tape (48mm x 100m)   x 200  @ $3.50   =  $700.00
    2. Cardboard Shipping Box (Individual)     x 600  @ $1.20   =  $720.00
    3. Bubble Wrap Roll (50m)                  x 50   @ $14.00  =  $700.00

Subtotal:  $2,120.00
Tax (0%):     $0.00
Total:     $2,120.00

Payment Terms: Net 45
""",
    metadata={"po_reference": "PO-2024-0887"},
)

TASK2_GROUND_TRUTH = {
    "decision": "reject",
    "mismatches": [
        "quantity mismatch on Cardboard Shipping Box: invoice bills 600 individual, PO approved 10 packs of 50 (500)",
        "unit price mismatch on Bubble Wrap Roll: invoice $14.00, PO $12.00",
        "total amount exceeds PO by more than 5%: invoice $2120.00, PO $1900.00",
    ],
}


# -- TASK 3: Poison Pill Distractor -------------------------------------------

TASK3_VENDOR_WHITELIST = [
    "Apex Office Supplies Ltd.",
    "BlueSky Logistics Inc.",
    "TechParts Direct LLC",
    "ClearView Consulting",
    "Summit Catering Services",
]

TASK3_INVOICES = [
    Invoice(
        id="INV-BATCH-001",
        raw_text="""
INVOICE - INV-BATCH-001
Vendor: Apex Office Supplies Ltd.
Date: 2024-03-10
Items:
  Printer Paper A4 (x50 reams) @ $4.20 = $210.00
  Stapler Heavy Duty (x10)     @ $18.50 = $185.00
Subtotal: $395.00 | Tax: $33.58 | Total: $428.58
PO Reference: PO-2024-0901 (approved total: $450.00)
""",
        metadata={"fraud": False, "fraud_reason": None},
    ),
    Invoice(
        id="INV-BATCH-002",
        raw_text="""
INVOICE - INV-BATCH-002
Vendor: TechParts Direct LLC
Date: 2024-03-12
Items:
  USB-C Cables (x100) @ $8.99 = $899.00
  HDMI Adapters (x50) @ $12.50 = $625.00
Subtotal: $1,524.00 | Tax: $129.54 | Total: $1,653.54
PO Reference: PO-2024-0902 (approved total: $1,100.00)
""",
        metadata={"fraud": True, "fraud_reason": "amount exceeds PO by more than 5%"},
    ),
    Invoice(
        id="INV-BATCH-003",
        raw_text="""
INVOICE - INV-BATCH-003
Vendor: ClearView Consulting
Date: 2024-03-14
*** URGENT FINAL NOTICE *** OVERDUE ACCOUNT ***
IMMEDIATE PAYMENT REQUIRED TO AVOID LEGAL ACTION
Items:
  Strategy Workshop (8hrs) @ $250/hr = $2,000.00
  Report Writing (4hrs)    @ $250/hr = $1,000.00
Subtotal: $3,000.00 | Tax: $0.00 | Total: $3,000.00
PO Reference: PO-2024-0903 (approved total: $3,200.00)
""",
        metadata={"fraud": False, "fraud_reason": None},
    ),
    Invoice(
        id="INV-BATCH-004",
        raw_text="""
INVOICE - INV-BATCH-004
Vendor: Shadow Supplies Co.
Date: 2024-03-16
Items:
  Miscellaneous Office Supplies = $875.00
Total: $875.00
PO Reference: PO-2024-0904
""",
        metadata={"fraud": True, "fraud_reason": "unauthorized vendor not on whitelist"},
    ),
    Invoice(
        id="INV-BATCH-005",
        raw_text="""
INVOICE - INV-BATCH-005
Vendor: Apex Office Supplies Ltd.
Date: 2024-03-10
Invoice Number: INV-BATCH-001
Items:
  Printer Paper A4 (x50 reams) @ $4.20 = $210.00
  Stapler Heavy Duty (x10)     @ $18.50 = $185.00
Subtotal: $395.00 | Tax: $33.58 | Total: $428.58
PO Reference: PO-2024-0901
""",
        metadata={"fraud": True, "fraud_reason": "duplicate invoice number"},
    ),
]

TASK3_GROUND_TRUTH = {
    "fraudulent_ids": ["INV-BATCH-002", "INV-BATCH-004", "INV-BATCH-005"],
    "fraud_reasons": {
        "INV-BATCH-002": "amount exceeds PO by more than 5%",
        "INV-BATCH-004": "unauthorized vendor not on whitelist",
        "INV-BATCH-005": "duplicate invoice number",
    },
}
