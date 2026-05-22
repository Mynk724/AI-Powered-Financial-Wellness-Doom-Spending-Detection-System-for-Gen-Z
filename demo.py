"""
demo.py — Synthetic data generator + FinWell AI demo runner
Run: python demo.py
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from finwell import FinWellAI


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

def generate_transactions(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate realistic Gen Z India spending transactions."""
    random.seed(seed)
    np.random.seed(seed)

    merchants_by_cat = {
        "food":          [("Swiggy", 280), ("Zomato", 320), ("McDonald's", 180),
                          ("Chai Point", 60), ("Starbucks", 350), ("Blinkit", 420)],
        "shopping":      [("Myntra", 1800), ("Ajio", 2200), ("Amazon", 1500),
                          ("Flipkart", 900), ("Nykaa", 650)],
        "entertainment": [("BookMyShow", 400), ("Steam", 1200), ("PVR Cinemas", 320)],
        "subscriptions": [("Netflix", 649), ("Spotify", 119), ("Hotstar", 299),
                          ("Adobe Creative", 1675), ("Notion Pro", 416)],
        "transport":     [("Uber", 180), ("Ola", 120), ("Rapido", 80), ("Metro", 50)],
        "utilities":     [("Jio Recharge", 239), ("Airtel", 399), ("Electricity Bill", 1100)],
        "health":        [("Apollo Pharmacy", 280), ("MedPlus", 150)],
    }

    impulse_notes = [
        "treat myself", "YOLO", "it was on sale", "needed this",
        "reward for working hard", "stress shopping", "feeling sad", "",
        "", "", "", "",  # mostly empty notes (realistic)
    ]

    rows = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(n):
        # 70% daytime, 30% late-night (realistic doom proportion)
        if random.random() < 0.28:
            hour = random.choice([22, 23, 0, 1, 2, 3])
        else:
            hour = random.randint(8, 21)

        day_offset = random.randint(0, 29)
        ts = base_date + timedelta(days=day_offset, hours=hour,
                                   minutes=random.randint(0, 59))

        # Pick category (weighted toward food & shopping for Gen Z)
        cat = random.choices(
            list(merchants_by_cat.keys()),
            weights=[30, 25, 10, 8, 15, 7, 5], k=1
        )[0]

        merchant, base_price = random.choice(merchants_by_cat[cat])
        # Add price variance ±30%
        amount = round(base_price * random.uniform(0.7, 1.3))

        # Occasional spending spike (doom events)
        if random.random() < 0.12:
            amount = round(amount * random.uniform(2.5, 4.0))

        rows.append({
            "id":        f"txn_{i:04d}",
            "amount":    amount,
            "merchant":  merchant,
            "timestamp": ts,
            "note":      random.choice(impulse_notes),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_subscriptions() -> list[dict]:
    """Sample subscription list with last-used dates."""
    today = datetime.now()
    return [
        {"name": "Spotify",        "amount": 119,  "last_used": today - timedelta(days=1),  "category": "music"},
        {"name": "Netflix",        "amount": 649,  "last_used": today - timedelta(days=3),  "category": "streaming"},
        {"name": "Hotstar Premium","amount": 299,  "last_used": today - timedelta(days=2),  "category": "streaming"},
        {"name": "Adobe Creative", "amount": 1675, "last_used": today - timedelta(days=18), "category": "design"},
        {"name": "Notion Pro",     "amount": 416,  "last_used": today - timedelta(days=35), "category": "productivity"},
        {"name": "Gym Membership", "amount": 700,  "last_used": today - timedelta(days=47), "category": "fitness"},
        {"name": "ChatGPT Plus",   "amount": 1680, "last_used": today - timedelta(days=0),  "category": "ai_tools"},
    ]


# ─────────────────────────────────────────────
# DEMO RUNNER
# ─────────────────────────────────────────────

def main():
    print("\n✨  Generating synthetic spending data...")
    df = generate_transactions(n=120)
    subscriptions = generate_subscriptions()

    print(f"   {len(df)} transactions | "
          f"₹{df['amount'].sum():,.0f} total spend | "
          f"date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    MONTHLY_INCOME = 75_000
    MONTHLY_BUDGET = 55_000

    engine = FinWellAI()
    results = engine.analyze(
        df=df,
        income=MONTHLY_INCOME,
        budget=MONTHLY_BUDGET,
        subscriptions=subscriptions,
    )

    engine.print_report(results)

    # Show a sample of detected doom transactions
    doom = results["transactions"][results["transactions"]["is_doom"]]
    print(f"Total doom transactions detected: {len(doom)} / {len(df)}")
    print(f"Doom spend total: ₹{doom['amount'].sum():,.0f}\n")


if __name__ == "__main__":
    main()
