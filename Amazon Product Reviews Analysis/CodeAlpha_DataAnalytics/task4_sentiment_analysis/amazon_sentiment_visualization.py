from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# =====================================
# Setup
# =====================================

nltk.download("vader_lexicon")

sns.set_style("whitegrid")

base_dir = Path(__file__).resolve().parent.parent

input_csv = base_dir / "datasets" / "amazon_reviews.csv"
output_csv = base_dir / "datasets" / "amazon_reviews_with_sentiment.csv"

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv(input_csv)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# =====================================
# Sentiment Analysis
# =====================================

sia = SentimentIntensityAnalyzer()


def classify_sentiment(text):
    if pd.isna(text):
        return "Neutral"

    score = sia.polarity_scores(str(text))
    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def compound_score(text):
    if pd.isna(text):
        return 0

    return sia.polarity_scores(str(text))["compound"]


df["Sentiment"] = df["review_text"].apply(classify_sentiment)
df["compound_score"] = df["review_text"].apply(compound_score)

# =====================================
# Review Length Feature
# =====================================

df["review_length"] = (
    df["review_text"]
    .astype(str)
    .apply(len)
)

# =====================================
# Save Dataset
# =====================================

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8"
)

print(
    f"\nSentiment dataset saved:\n{output_csv}"
)

# =====================================
# Chart 1
# Sentiment Distribution
# =====================================

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Sentiment",
    order=["Positive", "Neutral", "Negative"]
)

plt.title(
    "Sentiment Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sentiment")
plt.ylabel("Count")

for p in ax.patches:
    height = p.get_height()

    ax.annotate(
        f"{int(height)}",
        (
            p.get_x() + p.get_width() / 2,
            height
        ),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# =====================================
# Chart 2
# Sentiment Percentage
# =====================================

sentiment_pct = (
    df["Sentiment"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=sentiment_pct.index,
    y=sentiment_pct.values
)

plt.title(
    "Sentiment Percentage",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sentiment")
plt.ylabel("Percentage (%)")

for i, v in enumerate(sentiment_pct.values):
    ax.text(
        i,
        v + 0.5,
        f"{v:.1f}%",
        ha="center"
    )

plt.tight_layout()
plt.show()

# =====================================
# Chart 3
# Ratings Distribution
# =====================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="rating",
    order=[1, 2, 3, 4, 5]
)

plt.title(
    "Ratings Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Rating")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =====================================
# Chart 4
# Rating vs Sentiment
# =====================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="rating",
    hue="Sentiment",
    order=[1, 2, 3, 4, 5]
)

plt.title(
    "Rating vs Sentiment Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Rating")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# =====================================
# Chart 5
# Top Categories by Sentiment
# =====================================

category_sentiment = (
    df.groupby(
        ["category", "Sentiment"]
    )
    .size()
    .reset_index(name="count")
)

top_categories = (
    category_sentiment
    .sort_values(
        "count",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_categories,
    x="count",
    y="category",
    hue="Sentiment"
)

plt.title(
    "Top Categories by Sentiment",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Count")
plt.ylabel("Category")

plt.tight_layout()
plt.show()

# =====================================
# Chart 6
# Word Cloud
# =====================================

positive_text = " ".join(
    df[
        df["Sentiment"] == "Positive"
    ]["review_text"]
    .astype(str)
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(14, 7))

plt.imshow(
    wordcloud,
    interpolation="bilinear"
)

plt.axis("off")

plt.title(
    "Most Common Words in Positive Reviews",
    fontsize=16,
    fontweight="bold"
)

plt.show()

# =====================================
# Chart 7
# Review Length Analysis
# =====================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Sentiment",
    y="review_length"
)

plt.title(
    "Review Length by Sentiment",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sentiment")
plt.ylabel("Review Length")

plt.tight_layout()
plt.show()

# =====================================
# Chart 8
# Sentiment Score Distribution
# =====================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="compound_score",
    bins=30,
    kde=True
)

plt.title(
    "Sentiment Score Distribution",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Compound Score")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# =====================================
# Summary
# =====================================

print("\n========== SENTIMENT SUMMARY ==========\n")

print(
    df["Sentiment"]
    .value_counts()
)

print("\nAverage Rating:")
print(
    round(
        df["rating"].mean(),
        2
    )
)

print("\nAverage Review Length:")
print(
    round(
        df["review_length"].mean(),
        2
    )
)