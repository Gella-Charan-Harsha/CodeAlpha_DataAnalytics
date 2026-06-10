import csv
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
OUTPUT_CSV = "books.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}


@dataclass
class Book:
    title: str
    price: float
    rating: int


def parse_rating_text(rating_class: str) -> int:
    """Convert rating class like 'Three' to integer 3."""
    if not rating_class:
        return 0

    rating_word = rating_class.strip().split()[-1]
    mapping = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }
    return mapping.get(rating_word, 0)


def scrape_page(session: requests.Session, page_url: str) -> list[Book]:
    resp = session.get(page_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    books: list[Book] = []

    article_list = soup.select("article.product_pod")
    for article in article_list:
        # Title
        title = article.select_one("h3 a")
        title_text = str(title.get("title", "")).strip() if title else ""

        # Price
        price_tag = article.select_one("div.product_price p.price_color")
        price_text = price_tag.get_text(strip=True) if price_tag else ""
        # Example: "£51.77"
        # Handle potential encoding artifacts like 'Â51.77'
        cleaned = ''.join(ch for ch in price_text if ch.isdigit() or ch == '.')
        price_val = float(cleaned) if cleaned else 0.0

        # Rating
        rating_div = article.select_one("p.star-rating")
        rating_classes = rating_div.get("class") if rating_div else []
        rating_classes = rating_classes or []
        rating_val = parse_rating_text(" ".join(map(str, rating_classes)))

        # Always keep valid price/rating plus non-empty title.
        books.append(Book(title=title_text, price=price_val, rating=rating_val))

    return books


def main():
    output_path = OUTPUT_CSV

    with requests.Session() as session:
        all_books: list[Book] = []

        page_number = 1
        while True:
            if page_number == 1:
                page_url = urljoin(BASE_URL, "catalogue/page-1.html")
            else:
                page_url = urljoin(BASE_URL, f"catalogue/page-{page_number}.html")

            print(f"Scraping: {page_url}")
            try:
                page_books = scrape_page(session, page_url)
            except requests.HTTPError:
                break

            # If we fail to get products, stop.
            if not page_books:
                break

            all_books.extend(page_books)
            print(f"  Found {len(page_books)} books (total: {len(all_books)})")

            page_number += 1
            time.sleep(0.2)

    # Sort/clean and write
    all_books = all_books[:]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Price", "Rating"])
        for b in all_books:
            writer.writerow([b.title, b.price, b.rating])

    print(f"Saved {len(all_books)} rows to {output_path}")


if __name__ == "__main__":
    main()

