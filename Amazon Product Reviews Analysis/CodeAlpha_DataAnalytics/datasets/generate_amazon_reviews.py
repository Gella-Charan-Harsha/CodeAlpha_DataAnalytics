import csv
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# =============================
# Configuration
# =============================

CATEGORIES = [
    "Electronics",
    "Books",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Clothing",
]

# A small but varied product name library.
PRODUCT_TEMPLATES = {
    "Electronics": [
        "{brand} {model} Wireless Headphones",
        "{brand} {model} Smartwatch",
        "{brand} {model} Bluetooth Speaker",
        "{brand} {model} 4K Streaming Device",
        "{brand} {model} USB-C Charger (65W)",
    ],
    "Books": [
        "{author} - {title}",
        "{title} (Paperback) - {author}",
        "{title} - {author} (Kindle Edition)",
    ],
    "Home & Kitchen": [
        "{brand} {model} Air Fryer (5.3QT)",
        "{brand} {model} Coffee Grinder",
        "{brand} {model} Vacuum Insulated Tumbler (20oz)",
        "{brand} {model} Microfiber Sheet Set",
        "{brand} {model} Dish Rack Organizer",
    ],
    "Beauty": [
        "{brand} {model} Hydrating Face Serum",
        "{brand} {model} Anti-Frizz Hair Mask",
        "{brand} {model} SPF 50 Moisturizer",
        "{brand} {model} Gentle Exfoliating Cleanser",
        "{brand} {model} Vitamin C Brightening Cream",
    ],
    "Sports": [
        "{brand} {model} Fitness Resistance Bands",
        "{brand} {model} Breathable Running Shoes",
        "{brand} {model} Water Bottle (24oz)",
        "{brand} {model} Yoga Mat (6mm)",
        "{brand} {model} Sports Compression Socks",
    ],
    "Clothing": [
        "{brand} {model} Soft Cotton Hoodie",
        "{brand} {model} Performance T-Shirt",
        "{brand} {model} Everyday Ankle Socks (6-Pack)",
        "{brand} {model} Lightweight Jacket",
        "{brand} {model} Slim-Fit Jeans",
    ],
}

BRANDS = [
    "AuroTech",
    "Luminex",
    "EverWell",
    "Zenora",
    "NordPeak",
    "Brighton",
    "Cedar & Co.",
    "SonicWave",
    "CopperLeaf",
    "VivaStyle",
]

# Sentiment-aligned word banks.
POSITIVE_ADJECTIVES = [
    "excellent",
    "great",
    "fantastic",
    "amazing",
    "impressive",
    "solid",
    "well-made",
    "comfortable",
    "high-quality",
    "reliable",
    "durable",
    "superb",
]

NEGATIVE_ADJECTIVES = [
    "terrible",
    "awful",
    "poor",
    "disappointing",
    "weak",
    "flimsy",
    "difficult",
    "unreliable",
    "scratchy",
    "messy",
]

NEUTRAL_ADJECTIVES = [
    "decent",
    "average",
    "okay",
    "acceptable",
    "standard",
    "mixed",
    "fine",
]

POSITIVE_EXTRAS = [
    "The instructions were easy to follow.",
    "It arrived quickly and was well packaged.",
    "The build quality feels premium.",
    "I noticed an immediate improvement after using it.",
    "The design is thoughtful and looks great on my shelf.",
]

NEGATIVE_EXTRAS = [
    "However, the quality control seems inconsistent.",
    "The description is a bit misleading.",
    "I ran into issues after only a few days.",
    "The fit/material wasn\'t what I expected.",
    "Customer support took longer than I hoped.",
]

NEUTRAL_EXTRAS = [
    "It works as expected, but there\'s room for improvement.",
    "Overall it\'s fine for the price.",
    "Performance varies depending on usage.",
    "I\'d recommend it if you\'re on a budget.",
    "It delivered what was promised.",
]


POSITIVE_SENTENCE_PATTERNS = [
    "I\'m very happy with this {product}. {adjective_cap} and {feature}. {extra}",
    "This {product} exceeded my expectations. It\'s {adjective} and {feature}. {extra}",
    "Great value for the money. The {product} is {adjective} and feels {feature}. {extra}",
]

NEGATIVE_SENTENCE_PATTERNS = [
    "I\'m disappointed with this {product}. It\'s {adjective} and {feature}. {extra}",
    "This {product} didn\'t work out for me. The build felt {adjective} and {feature}. {extra}",
    "Not worth the price. The {product} is {adjective} and {feature}. {extra}",
]

NEUTRAL_SENTENCE_PATTERNS = [
    "This {product} is {adjective}. It\'s {feature} for everyday use. {extra}",
    "My experience with the {product} is {adjective}. It\'s {feature}. {extra}",
    "This {product} is {adjective} overall. You\'ll get what you pay for—{feature}. {extra}",
]


FEATURES_BY_CATEGORY = {
    "Electronics": [
        "the sound is clear",
        "it connects quickly",
        "battery life is solid",
        "the setup is straightforward",
        "the display is crisp",
        "the buttons feel responsive",
    ],
    "Books": [
        "the writing style is engaging",
        "the pacing is smooth",
        "the content is well organized",
        "the story keeps you hooked",
        "the examples are helpful",
        "the formatting is easy to read",
    ],
    "Home & Kitchen": [
        "it cooks evenly",
        "it\'s easy to clean",
        "the materials feel sturdy",
        "it saves counter space",
        "it works efficiently",
        "it holds up after multiple uses",
    ],
    "Beauty": [
        "it absorbs quickly",
        "my skin feels softer",
        "it doesn\'t irritate easily",
        "it leaves a nice finish",
        "the scent is pleasant",
        "it moisturizes well",
    ],
    "Sports": [
        "it provides good support",
        "it\'s breathable",
        "it feels comfortable during workouts",
        "it\'s grippy on the mat",
        "it\'s easy to adjust",
        "it\'s durable for training",
    ],
    "Clothing": [
        "the fabric feels soft",
        "it fits true to size",
        "the stitching looks durable",
        "it\'s comfortable for all-day wear",
        "the material breathes well",
        "it looks exactly like the photos",
    ],
}


RATING_TO_SENTIMENT = {
    # 1-2 => Negative, 3 => Neutral, 4-5 => Positive
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive",
}


def _random_date(start_days_ago: int = 365, end_days_ago: int = 0) -> str:
    today = datetime.now().date()
    lo, hi = sorted((start_days_ago, end_days_ago))
    delta_days = random.randint(lo, hi)
    d = today - timedelta(days=delta_days)
    return d.strftime("%Y-%m-%d")



def _safe_text(s: str) -> str:
    # Keep it simple: remove excessive whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def generate_one_review(category: str, sentiment: str, rating: int, used_texts: set[str]) -> dict:
    brand = random.choice(BRANDS)
    model = random.choice(
        [
            "X1",
            "A7",
            "Pro",
            "Plus",
            "Series 2",
            "Lite",
            "Max",
            "V2",
            "SE",
            "Air",
        ]
    )

    adjective = random.choice(
        POSITIVE_ADJECTIVES
        if sentiment == "Positive"
        else NEGATIVE_ADJECTIVES
        if sentiment == "Negative"
        else NEUTRAL_ADJECTIVES
    )

    adjective_cap = adjective[:1].upper() + adjective[1:]

    feature = random.choice(FEATURES_BY_CATEGORY[category])

    extra = random.choice(
        POSITIVE_EXTRAS
        if sentiment == "Positive"
        else NEGATIVE_EXTRAS
        if sentiment == "Negative"
        else NEUTRAL_EXTRAS
    )

    # Build product name.
    product_template = random.choice(PRODUCT_TEMPLATES[category])
    author = random.choice(["Jordan Maxwell", "Emily Carter", "Ravi Patel", "Sofia Nguyen", "Noah Brooks"])
    title = random.choice([
        "The Practical Guide to Everyday Wins",
        "Beyond the Baseline",
        "Designing Calm",
        "The Quiet Strategy",
        "A Better Morning",
        "Foundations of Focus",
        "Small Steps, Big Results",
        "Stories from the Workshop",
    ])

    product_name = product_template.format(brand=brand, model=model, author=author, title=title)

    templates = (
        POSITIVE_SENTENCE_PATTERNS
        if sentiment == "Positive"
        else NEGATIVE_SENTENCE_PATTERNS
        if sentiment == "Negative"
        else NEUTRAL_SENTENCE_PATTERNS
    )

    sentence = random.choice(templates).format(
        product=product_name,
        adjective=adjective,
        adjective_cap=adjective_cap,
        feature=feature,
        extra=extra,
    )

    # Variable length: add another sentence 0-2 times for longer reviews.
    length_mode = random.choices(["short", "medium", "detailed"], weights=[0.35, 0.40, 0.25])[0]
    sentences = [sentence]
    if length_mode in {"medium", "detailed"}:
        sentences.append(
            _safe_text(
                random.choice(
                    [
                        f"I\'ve been using this for a bit and it feels {adjective} overall.",
                        "The value seems to match the description.",
                        "It\'s practical and easy to work with.",
                        "The details are where it stands out.",
                    ]
                )
            )
        )
    if length_mode == "detailed":
        sentences.append(
            _safe_text(
                random.choice(
                    [
                        "After a few uses, I\'m noticing consistent results.",
                        "There were a couple small quirks, but nothing deal-breaking for me.",
                        "If you\'re considering it, I think you\'ll be happy with the everyday performance.",
                        "I appreciate the design choices; they make day-to-day use smoother.",
                    ]
                )
            )
        )

    review_text = " ".join(sentences)

    # Avoid duplicates by text signature.
    signature = review_text.lower()
    if signature in used_texts:
        return generate_one_review(category, sentiment, rating, used_texts)
    used_texts.add(signature)

    return {
        "review_id": str(uuid.uuid4()),
        "product_name": product_name,
        "category": category,
        "rating": rating,
        "review_text": review_text,
        "review_date": _random_date(),
    }


def generate_dataset(output_csv: str, rows: int = 5000, seed: int = 42) -> None:
    random.seed(seed)

    # Balanced sentiment distribution.
    sentiments = [
        ("Positive", 0.40),
        ("Neutral", 0.20),
        ("Negative", 0.40),
    ]

    sentiment_choices = [s for s, _ in sentiments]
    sentiment_weights = [w for _, w in sentiments]

    # Rating selection aligned with sentiment.
    rating_options = {
        "Positive": [4, 5],
        "Neutral": [3],
        "Negative": [1, 2],
    }

    used_texts: set[str] = set()

    categories_weighted = [
        "Electronics",
        "Books",
        "Home & Kitchen",
        "Beauty",
        "Sports",
        "Clothing",
    ]

    category_weights = [0.18, 0.18, 0.18, 0.15, 0.15, 0.16]

    records: list[dict] = []

    for _ in range(rows):
        category = random.choices(categories_weighted, weights=category_weights, k=1)[0]
        sentiment = random.choices(sentiment_choices, weights=sentiment_weights, k=1)[0]

        # Slight randomization for sentiment ratings.
        rating = random.choice(rating_options[sentiment])

        records.append(generate_one_review(category, sentiment, rating, used_texts))

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_id",
                "product_name",
                "category",
                "rating",
                "review_text",
                "review_date",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def run_sentiment_and_visualize(
    input_csv: str,
    output_csv: str,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from wordcloud import WordCloud


    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    df = pd.read_csv(input_csv)

    def classify_vader(text: str) -> str:
        if pd.isna(text):
            return "Neutral"
        scores = sia.polarity_scores(str(text))
        compound = scores.get("compound", 0.0)
        if compound >= 0.05:
            return "Positive"
        if compound <= -0.05:
            return "Negative"
        return "Neutral"

    df["Sentiment"] = df["review_text"].apply(classify_vader)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    # =====================
    # Visualizations
    # =====================
    sns.set_style("whitegrid")

    sentiment_order = ["Positive", "Neutral", "Negative"]

    # 1) Sentiment Distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="Sentiment", order=sentiment_order)
    plt.title("Sentiment Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )
    plt.tight_layout()
    plt.show()

    # 2) Sentiment Percentage
    plt.figure(figsize=(8, 5))
    sentiment_pct = df["Sentiment"].value_counts(normalize=True).reindex(sentiment_order) * 100
    ax = sns.barplot(x=sentiment_pct.index, y=sentiment_pct.values, palette=["green", "gray", "red"])
    plt.title("Sentiment Percentage", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Percentage (%)")
    for i, v in enumerate(sentiment_pct.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()

    # 3) Ratings Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="rating", order=[1, 2, 3, 4, 5])
    plt.title("Ratings Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 4) Rating vs Sentiment Analysis
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="rating", hue="Sentiment", order=[1, 2, 3, 4, 5])
    plt.title("Rating vs Sentiment Analysis", fontsize=16, fontweight="bold")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.show()

    # 5) Top Categories by Sentiment
    plt.figure(figsize=(10, 6))
    top_cats = (
        df.groupby(["category", "Sentiment"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    # Take top 10 category-sentiment combinations
    top_cats = top_cats.head(10)
    sns.barplot(data=top_cats, x="count", y="category", hue="Sentiment", orient="h")
    plt.title("Top Categories by Sentiment (Top 10)", fontsize=16, fontweight="bold")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()

    # Review length and compound score features
    df["review_length"] = df["review_text"].astype(str).apply(len)
    df["compound_score"] = df["review_text"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    # 6) Review Length by Sentiment
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="Sentiment",
        y="review_length",
    )
    plt.title("Review Length by Sentiment", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Review Length")
    plt.tight_layout()
    plt.show()

    # 7) Sentiment Score Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="compound_score",
        bins=30,
        kde=True,
    )
    plt.title("Sentiment Score Distribution", fontsize=16, fontweight="bold")
    plt.xlabel("Compound Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 8) Word Cloud (Positive Reviews)
    positive_text = " ".join(
        df[df["Sentiment"] == "Positive"]["review_text"].astype(str)
    )
    if positive_text.strip():
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
        ).generate(positive_text)

        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            "Most Common Words in Positive Reviews",
            fontsize=16,
            fontweight="bold",
        )
        plt.show()

    print("\n========== SENTIMENT ANALYSIS SUMMARY ==========")
    print("\nDataset Shape:")
    print(df.shape)
    print("\nSentiment Distribution:")
    print(df["Sentiment"].value_counts())
    print("\nCategory Distribution:")
    print(df["category"].value_counts())
    print("\nAverage Rating:")
    print(round(df["rating"].mean(), 2))
    print("\nAverage Review Length:")
    print(round(df["review_length"].mean(), 2))



def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_csv = str(base_dir / "amazon_reviews.csv")
    dataset_with_sentiment_csv = str(base_dir / "amazon_reviews_with_sentiment.csv")

    # Generate a larger dataset (>= 5000).
    generate_dataset(dataset_csv, rows=5000, seed=42)

    # Run sentiment analysis and produce output CSV.
    run_sentiment_and_visualize(dataset_csv, dataset_with_sentiment_csv)

    print(f"Saved dataset: {dataset_csv}")
    print(f"Saved dataset with sentiment: {dataset_with_sentiment_csv}")


if __name__ == "__main__":
    main()

