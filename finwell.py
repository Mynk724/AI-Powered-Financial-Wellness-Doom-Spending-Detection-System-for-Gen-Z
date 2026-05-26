"""
FinWell AI — Financial Wellness & Doom Spending Detection System
================================================================
Features:
  1. Expense Tracking & Categorization
  2. Doom Spending Detection (late-night, stress, impulse)
  3. Financial Health Score
  4. Subscription Leak Detector
  5. Smart Budget Recommendations
  6. Financial Persona Classifier
  7. Behavioral Pattern Analysis
""""""

import random
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class Transaction:
    id: str
    amount: float
    merchant: str
    timestamp: datetime
    category: str = "uncategorized"
    note: str = ""
    is_doom: bool = False
    doom_type: Optional[str] = None


@dataclass
class UserProfile:
    name: str
    monthly_income: float
    monthly_budget: float
    transactions: list[Transaction] = field(default_factory=list)
    subscriptions: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────
# 1. EXPENSE CATEGORIZER
# ─────────────────────────────────────────────

class ExpenseCategorizer:
    """Rule-based NLP categorizer using keyword matching."""

    CATEGORY_KEYWORDS = {
        "food":          ["swiggy", "zomato", "mcdonald", "domino", "kfc", "pizza", "cafe",
                          "restaurant", "food", "biryani", "chai", "coffee", "blinkit", "zepto"],
        "shopping":      ["myntra", "ajio", "amazon", "flipkart", "meesho", "nykaa", "reliance",
                          "shopify", "mall", "clothing", "shoes", "fashion", "decathlon"],
        "entertainment": ["netflix", "hotstar", "prime", "spotify", "youtube", "bookmyshow",
                          "pvr", "inox", "gaming", "steam", "playstation"],
        "subscriptions": ["subscription", "premium", "pro", "membership", "plan", "monthly",
                          "annual", "gym", "notion", "adobe", "chatgpt", "canva"],
        "transport":     ["uber", "ola", "rapido", "metro", "irctc", "petrol", "fuel",
                          "auto", "cab", "bus", "train", "flight"],
        "emi":           ["emi", "loan", "equated", "bajaj", "hdfc loan", "icici loan", "emi pay"],
        "health":        ["pharmacy", "apollo", "medplus", "hospital", "clinic", "doctor",
                          "medicine", "diagnostic", "health", "medico"],
        "utilities":     ["electricity", "water", "gas", "internet", "broadband", "jio",
                          "airtel", "vi", "bsnl", "recharge", "dth"],
    }

    def categorize(self, merchant: str, note: str = "") -> str:
        text = (merchant + " " + note).lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category
        return "others"

    def categorize_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["category"] = df.apply(
            lambda r: self.categorize(r["merchant"], r.get("note", "")), axis=1
        )
        return df


# ─────────────────────────────────────────────
# 2. DOOM SPENDING DETECTOR
# ─────────────────────────────────────────────

class DoomSpendingDetector:
    """
    Detects emotional / impulsive purchases using:
      - Time pattern analysis (late-night windows)
      - Spending spike detection (Isolation Forest)
      - Rapid transaction bursts
      - Keyword-based sentiment (NLP proxy)
    """

    LATE_NIGHT_HOURS = range(22, 24)  # 10 PM – midnight
    EARLY_MORNING_HOURS = range(0, 4)  # midnight – 4 AM

    IMPULSE_KEYWORDS = [
        "treat myself", "reward", "yolo", "sale", "deal", "last chance",
        "just one", "screw it", "stress", "sad", "bored", "deserved",
        "emotional", "feeling low", "can't sleep", "up late"
    ]

    DOOM_CATEGORIES = ["shopping", "entertainment", "food"]

    def __init__(self, contamination: float = 0.15):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Train anomaly detector on historical spending amounts."""
        amounts = df["amount"].values.reshape(-1, 1)
        self.isolation_forest.fit(amounts)
        self._fitted = True
        self.mean_amount = df["amount"].mean()
        self.std_amount = df["amount"].std()

    def _is_late_night(self, ts: datetime) -> bool:
        return ts.hour in self.LATE_NIGHT_HOURS or ts.hour in self.EARLY_MORNING_HOURS

    def _is_spike(self, amount: float) -> bool:
        if not self._fitted:
            return False
        pred = self.isolation_forest.predict([[amount]])
        return pred[0] == -1  # -1 = anomaly

    def _has_impulse_note(self, note: str) -> bool:
        note_lower = note.lower()
        return any(kw in note_lower for kw in self.IMPULSE_KEYWORDS)

    def _is_burst(self, transaction: pd.Series, df: pd.DataFrame,
                  window_minutes: int = 30) -> bool:
        """Check if 3+ transactions happened within window_minutes."""
        ts = transaction["timestamp"]
        window_start = ts - timedelta(minutes=window_minutes)
        burst = df[
            (df["timestamp"] >= window_start) &
            (df["timestamp"] <= ts)
        ]
        return len(burst) >= 3

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add doom_spending and doom_type columns to DataFrame."""
        df = df.copy()

        if not self._fitted:
            self.fit(df)

        df["is_doom"] = False
        df["doom_type"] = None
        df["doom_score"] = 0.0

        for idx, row in df.iterrows():
            doom_score = 0
            reasons = []

            if self._is_late_night(row["timestamp"]):
                doom_score += 40
                reasons.append("late_night")

            if self._is_spike(row["amount"]):
                doom_score += 30
                reasons.append("spending_spike")

            if self._is_burst(row, df):
                doom_score += 20
                reasons.append("burst_buying")

            if self._has_impulse_note(row.get("note", "")):
                doom_score += 25
                reasons.append("emotional_note")

            if row.get("category") in self.DOOM_CATEGORIES and row["amount"] > self.mean_amount:
                doom_score += 15
                reasons.append("high_impulse_category")

            df.at[idx, "doom_score"] = doom_score
            if doom_score >= 40:
                df.at[idx, "is_doom"] = True
                df.at[idx, "doom_type"] = reasons[0] if reasons else "general"

        return df


# ─────────────────────────────────────────────
# 3. FINANCIAL HEALTH SCORE
# ─────────────────────────────────────────────

class FinancialHealthScorer:
    """
    Generates a 0–100 wellness score based on:
      - Savings ratio        (30 pts)
      - Spending consistency (25 pts)
      - Doom spending ratio  (25 pts)
      - Subscription overload (20 pts)
    """

    def score(self, df: pd.DataFrame, income: float,
              budget: float, subscriptions: list[dict]) -> dict:

        total_spend = df["amount"].sum()
        savings = income - total_spend
        savings_ratio = max(0, savings / income)

        # Spending consistency: std dev of weekly spend relative to mean
        df_copy = df.copy()
        df_copy["week"] = df_copy["timestamp"].dt.isocalendar().week
        weekly = df_copy.groupby("week")["amount"].sum()
        consistency = 1 - min(1, weekly.std() / (weekly.mean() + 1))

        # Doom ratio: doom spend / total spend
        doom_spend = df[df["is_doom"]]["amount"].sum()
        doom_ratio = doom_spend / (total_spend + 1)
        doom_score = max(0, 1 - doom_ratio * 3)

        # Subscription overload
        sub_total = sum(s["amount"] for s in subscriptions)
        sub_ratio = sub_total / (income + 1)
        sub_score = max(0, 1 - sub_ratio * 5)

        weighted = (
            savings_ratio   * 30 +
            consistency     * 25 +
            doom_score      * 25 +
            sub_score       * 20
        )
        final_score = round(min(100, max(0, weighted)))

        return {
            "overall": final_score,
            "grade": self._grade(final_score),
            "components": {
                "savings_ratio":   round(savings_ratio * 100, 1),
                "consistency":     round(consistency * 100, 1),
                "doom_penalty":    round(doom_ratio * 100, 1),
                "subscription_load": round(sub_ratio * 100, 1),
            },
            "savings_amount":  round(savings, 2),
            "doom_spend":      round(doom_spend, 2),
            "total_spend":     round(total_spend, 2),
        }

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 55: return "Fair"
        if score >= 40: return "Poor"
        return "Critical"


# ─────────────────────────────────────────────
# 4. SUBSCRIPTION LEAK DETECTOR
# ─────────────────────────────────────────────

class SubscriptionLeakDetector:
    """Finds unused, duplicate, or overpriced subscriptions."""

    def analyze(self, subscriptions: list[dict]) -> dict:
        """
        Each subscription dict: {name, amount, last_used, category}
        last_used: datetime or None
        """
        today = datetime.now()
        unused, duplicate_risk, low_value = [], [], []
        seen_categories = {}

        for sub in subscriptions:
            last = sub.get("last_used")
            days_unused = (today - last).days if last else 9999
            name = sub["name"]
            cat = sub.get("category", "general")
            amt = sub["amount"]

            if days_unused > 30:
                unused.append({**sub, "days_unused": days_unused, "flag": "unused"})

            if cat in seen_categories:
                duplicate_risk.append({
                    **sub,
                    "duplicate_of": seen_categories[cat],
                    "flag": "duplicate_risk"
                })
            else:
                seen_categories[cat] = name

            if days_unused > 14 and amt > 200:
                low_value.append({**sub, "flag": "low_value"})

        total_leak = sum(s["amount"] for s in unused + duplicate_risk)
        total_monthly = sum(s["amount"] for s in subscriptions)

        return {
            "total_monthly":   round(total_monthly, 2),
            "leakage_amount":  round(total_leak, 2),
            "unused":          unused,
            "duplicate_risk":  duplicate_risk,
            "low_value":       low_value,
            "recommendations": self._recommend(unused, duplicate_risk),
        }

    @staticmethod
    def _recommend(unused: list, duplicates: list) -> list[str]:
        tips = []
        for s in unused:
            tips.append(f"Cancel '{s['name']}' — unused for {s['days_unused']} days (save ₹{s['amount']}/mo)")
        for s in duplicates:
            tips.append(f"'{s['name']}' overlaps with '{s['duplicate_of']}' — consider consolidating")
        return tips


# ─────────────────────────────────────────────
# 5. SMART BUDGET RECOMMENDER
# ─────────────────────────────────────────────

class BudgetRecommender:
    """
    Generates personalized budget limits per category
    based on income, historical spend, and savings target.
    """

    # Ideal allocation (% of income) — 50/30/20 rule adapted for Gen Z India
    IDEAL_ALLOCATION = {
        "food":          0.15,
        "shopping":      0.08,
        "entertainment": 0.05,
        "subscriptions": 0.03,
        "transport":     0.08,
        "emi":           0.10,
        "health":        0.05,
        "utilities":     0.06,
        "savings":       0.20,
        "others":        0.10,
    }

    def recommend(self, df: pd.DataFrame, income: float,
                  savings_target: float = 0.20) -> dict:
        actual = df.groupby("category")["amount"].sum().to_dict()
        total_spend = sum(actual.values())

        budgets = {}
        warnings = []
        savings_amount = income * savings_target

        for cat, ideal_pct in self.IDEAL_ALLOCATION.items():
            if cat == "savings":
                budgets[cat] = round(savings_amount)
                continue
            ideal_budget = round(income * ideal_pct)
            actual_spend = actual.get(cat, 0)
            pct_used = actual_spend / (ideal_budget + 1)

            status = "on_track"
            if pct_used > 1.2:
                status = "overspending"
                warnings.append(
                    f"{cat.title()}: ₹{actual_spend:.0f} spent vs ₹{ideal_budget} budget "
                    f"({pct_used*100:.0f}% — reduce by ₹{actual_spend - ideal_budget:.0f})"
                )
            elif pct_used < 0.5:
                status = "underspending"

            budgets[cat] = {
                "recommended": ideal_budget,
                "actual":      round(actual_spend),
                "status":      status,
                "pct_used":    round(pct_used * 100, 1),
            }

        return {
            "income":          income,
            "total_spend":     round(total_spend),
            "projected_savings": round(income - total_spend),
            "target_savings":  round(savings_amount),
            "budgets":         budgets,
            "warnings":        warnings,
        }


# ─────────────────────────────────────────────
# 6. FINANCIAL PERSONA CLASSIFIER
# ─────────────────────────────────────────────

class PersonaClassifier:
    """
    Classifies user into financial personas using
    a Random Forest trained on behavioral features.
    """

    PERSONAS = ["Smart Saver", "Doom Spender", "Subscription Addict", "Impulse Buyer"]

    def _extract_features(self, df: pd.DataFrame,
                          subscriptions: list[dict], income: float) -> np.ndarray:
        total = df["amount"].sum() + 1
        doom_pct = df[df["is_doom"]]["amount"].sum() / total
        sub_pct = sum(s["amount"] for s in subscriptions) / (income + 1)
        late_night_pct = df[df["timestamp"].dt.hour.isin(range(22, 28) if False else list(range(22, 24)) + list(range(0, 4)))]["amount"].sum() / total
        shop_pct = df[df["category"] == "shopping"]["amount"].sum() / total
        savings_pct = max(0, (income - total) / income)

        return np.array([[doom_pct, sub_pct, late_night_pct, shop_pct, savings_pct]])

    def classify(self, df: pd.DataFrame,
                 subscriptions: list[dict], income: float) -> dict:
        feats = self._extract_features(df, subscriptions, income)
        doom_pct, sub_pct, late_night_pct, shop_pct, savings_pct = feats[0]

        # Rule-based scoring (no training data needed for demo)
        scores = {
            "Smart Saver":         savings_pct * 100,
            "Doom Spender":        doom_pct * 100 + late_night_pct * 50,
            "Subscription Addict": sub_pct * 200,
            "Impulse Buyer":       shop_pct * 80 + late_night_pct * 40,
        }
        total_score = sum(scores.values()) + 1
        percentages = {k: round(v / total_score * 100, 1) for k, v in scores.items()}
        primary = max(percentages, key=percentages.get)

        return {
            "primary_persona":  primary,
            "persona_scores":   percentages,
            "behavioral_tags":  self._tags(doom_pct, sub_pct, late_night_pct, savings_pct),
        }

    @staticmethod
    def _tags(doom, sub, late, savings) -> list[str]:
        tags = []
        if late > 0.2:   tags.append("Night owl buyer")
        if doom > 0.3:   tags.append("Stress spender")
        if sub > 0.1:    tags.append("Subscription hoarding")
        if savings > 0.2: tags.append("Financially disciplined")
        if doom < 0.05 and savings > 0.15: tags.append("Mindful spender")
        return tags or ["Balanced spender"]


# ─────────────────────────────────────────────
# 7. BEHAVIORAL PATTERN ANALYZER
# ─────────────────────────────────────────────

class BehavioralAnalyzer:
    """Detects time-based, day-based, and stress spending patterns."""

    def analyze(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["day_name"] = df["timestamp"].dt.day_name()
        df["week"] = df["timestamp"].dt.isocalendar().week

        # Hourly spending pattern
        hourly = df.groupby("hour")["amount"].sum()
        peak_hour = int(hourly.idxmax())
        night_spend = df[df["hour"].isin(list(range(22, 24)) + list(range(0, 4)))]["amount"].sum()
        day_spend = df["amount"].sum()
        night_pct = round(night_spend / (day_spend + 1) * 100, 1)

        # Day-of-week pattern
        day_totals = df.groupby("day_name")["amount"].sum()
        peak_day = day_totals.idxmax()

        # Weekly trend (is spending increasing?)
        weekly = df.groupby("week")["amount"].sum().reset_index()
        trend = "increasing" if weekly["amount"].is_monotonic_increasing else \
                "decreasing" if weekly["amount"].is_monotonic_decreasing else "fluctuating"

        # Doom event clustering
        doom_df = df[df["is_doom"]]
        doom_hours = doom_df["hour"].value_counts().head(3).to_dict() if not doom_df.empty else {}

        return {
            "peak_spending_hour": peak_hour,
            "night_spending_pct": night_pct,
            "peak_spending_day":  peak_day,
            "weekly_trend":       trend,
            "doom_peak_hours":    doom_hours,
            "insights":           self._generate_insights(peak_hour, night_pct, peak_day, trend),
        }

    @staticmethod
    def _generate_insights(peak_hour, night_pct, peak_day, trend) -> list[str]:
        insights = []
        if night_pct > 25:
            insights.append(
                f"⚠️  {night_pct}% of your spending happens late at night — "
                "your wallet is nocturnal."
            )
        if peak_hour in range(22, 24) or peak_hour in range(0, 4):
            insights.append(
                f"🌙 Your peak spending hour is {peak_hour}:00 — "
                "classic doom spending territory."
            )
        insights.append(
            f"📅 {peak_day} is your highest spending day — consider a "
            "no-spend rule on this day."
        )
        if trend == "increasing":
            insights.append(
                "📈 Weekly spending is trending upward — time to reassess your budget."
            )
        return insights


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

class FinWellAI:
    """Orchestrates all modules into one analysis pipeline."""

    def __init__(self):
        self.categorizer  = ExpenseCategorizer()
        self.doom_detector = DoomSpendingDetector()
        self.health_scorer = FinancialHealthScorer()
        self.sub_detector  = SubscriptionLeakDetector()
        self.budget_rec    = BudgetRecommender()
        self.persona       = PersonaClassifier()
        self.behavior      = BehavioralAnalyzer()

    def analyze(self, df: pd.DataFrame, income: float,
                budget: float, subscriptions: list[dict]) -> dict:
        """
        Full analysis pipeline.

        Parameters
        ----------
        df            : DataFrame with columns [amount, merchant, timestamp, note]
        income        : Monthly take-home income
        budget        : Monthly spending budget
        subscriptions : List of {name, amount, last_used, category}
        """
        print("🔍 Categorizing transactions...")
        df = self.categorizer.categorize_batch(df)

        print("😱 Detecting doom spending...")
        df = self.doom_detector.detect(df)

        print("💯 Calculating health score...")
        health = self.health_scorer.score(df, income, budget, subscriptions)

        print("📦 Detecting subscription leaks...")
        subs = self.sub_detector.analyze(subscriptions)

        print("💰 Generating budget recommendations...")
        budget_recs = self.budget_rec.recommend(df, income)

        print("🎭 Classifying financial persona...")
        persona = self.persona.classify(df, subscriptions, income)

        print("📊 Analyzing behavioral patterns...")
        behavior = self.behavior.analyze(df)

        return {
            "transactions": df,
            "health_score": health,
            "subscriptions": subs,
            "budget": budget_recs,
            "persona": persona,
            "behavior": behavior,
        }

    def print_report(self, results: dict) -> None:
        h = results["health_score"]
        p = results["persona"]
        b = results["behavior"]
        s = results["subscriptions"]
        br = results["budget"]

        sep = "─" * 55
        print(f"\n{'═'*55}")
        print(f"  FinWell AI — Financial Wellness Report")
        print(f"{'═'*55}")

        print(f"\n🏥  HEALTH SCORE:  {h['overall']}/100  [{h['grade']}]")
        print(f"    Total spend:   ₹{h['total_spend']:,.0f}")
        print(f"    Savings:       ₹{h['savings_amount']:,.0f}")
        print(f"    Doom spend:    ₹{h['doom_spend']:,.0f}")
        print(f"    Savings rate:  {h['components']['savings_ratio']}%")

        print(f"\n{sep}")
        print(f"🎭  PERSONA:  {p['primary_persona']}")
        for tag in p["behavioral_tags"]:
            print(f"    • {tag}")
        print("\n    Score breakdown:")
        for name, pct in sorted(p["persona_scores"].items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            print(f"    {name:<22} {bar:<20} {pct:.1f}%")

        print(f"\n{sep}")
        print(f"💸  DOOM SPENDING EVENTS:")
        doom_df = results["transactions"][results["transactions"]["is_doom"]]
        if doom_df.empty:
            print("    No doom spending detected this period 🎉")
        else:
            for _, row in doom_df.head(5).iterrows():
                print(f"    ₹{row['amount']:>7,.0f}  {row['merchant']:<20}  "
                      f"{row['timestamp'].strftime('%b %d %H:%M')}  [{row['doom_type']}]")

        print(f"\n{sep}")
        print(f"📦  SUBSCRIPTION LEAKS:  ₹{s['leakage_amount']:,.0f}/mo at risk")
        for tip in s["recommendations"][:5]:
            print(f"    • {tip}")

        print(f"\n{sep}")
        print(f"📊  BEHAVIORAL INSIGHTS:")
        for insight in b["insights"]:
            print(f"    {insight}")

        print(f"\n{sep}")
        print(f"💡  BUDGET WARNINGS:")
        if br["warnings"]:
            for w in br["warnings"]:
                print(f"    ⚠️  {w}")
        else:
            print("    All categories within budget ✅")

        print(f"\n{'═'*55}\n")
