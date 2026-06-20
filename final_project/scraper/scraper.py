"""
scraper.py - Zameen.com Islamabad House Scraper
=============================================================================
SCRAPES: House ONLY
SPEED:   Increased workers + faster timeouts + batch processing

SETUP:
    pip install selenium webdriver-manager beautifulsoup4 pandas requests

RUN:
    python scraper.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import requests
import time, random, os, json

# ── Config ────────────────────────────────────────────────────
OUTPUT_CSV     = "../data/zameen_islamabad.csv"
TARGET_COUNT   = 5000
MAX_PAGES      = 250
DETAIL_WORKERS = 20
REQUEST_TIMEOUT= 8
PAGE_WAIT_MIN  = 3.0
PAGE_WAIT_MAX  = 5.0
JITTER_MIN     = 0.2
JITTER_MAX     = 0.8

ALLOWED_TYPES  = {"House"}
os.makedirs("../data", exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── Requests session ──────────────────────────────────────────
def make_session():
    s = requests.Session()
    retry = Retry(
        total=2, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adp = HTTPAdapter(max_retries=retry, pool_connections=30, pool_maxsize=30)
    s.mount("https://", adp)
    s.mount("http://",  adp)
    s.headers.update({
        "User-Agent":      UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer":         "https://www.zameen.com/",
        "Connection":      "keep-alive",
    })
    return s

SESSION = make_session()

# ── Selenium driver ───────────────────────────────────────────
def get_driver():
    opt = webdriver.ChromeOptions()
    opt.add_argument("--headless")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--disable-blink-features=AutomationControlled")
    opt.add_argument("--disable-extensions")
    opt.add_argument("--blink-settings=imagesEnabled=false")
    opt.add_argument(f"user-agent={UA}")
    opt.add_experimental_option("excludeSwitches", ["enable-automation"])
    opt.add_experimental_option("useAutomationExtension", False)
    opt.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
    })
    opt.page_load_strategy = "eager"
    drv = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opt
    )
    drv.set_page_load_timeout(25)
    return drv


# ═══════════════════════════════════════════════════════════════
#  DETAIL PAGE PARSING
# ═══════════════════════════════════════════════════════════════

EMPTY_EXTRAS = {
    "built_in_year": None, "parking": None,
    "servant_quarters": None, "store_rooms": None,
    "kitchens": None, "drawing_rooms": None,
}

LABEL_PATTERNS = {
    "built_in_year":    ["built in year", "year built", "construction year"],
    "parking":          ["parking spaces", "car parking", "parking"],
    "servant_quarters": ["servant quarters", "servants quarters", "servant quarter"],
    "store_rooms":      ["store rooms", "store room", "storerooms"],
    "kitchens":         ["kitchens", "kitchen"],
    "drawing_rooms":    ["drawing rooms", "drawing room"],
}

def safe_year(val):
    try:
        v = int(str(val).strip())
        return v if 1950 <= v <= 2026 else None
    except:
        return None

def safe_count(val, cap=10):
    try:
        v = int(str(val).strip())
        return v if 0 <= v <= cap else None
    except:
        if isinstance(val, str) and any(w in val.lower() for w in ["yes", "available"]):
            return 1
        return None

def parse_colon_text(text, key):
    if ":" not in text:
        return None
    value_part = text.split(":", 1)[1].strip()
    if not value_part:
        return None
    if key == "built_in_year":
        return safe_year(value_part)
    return safe_count(value_part, cap=5 if key != "parking" else 10)

def parse_amenities(soup):
    result = dict(EMPTY_EXTRAS)
    all_span_texts = []
    for li in soup.find_all("li"):
        for span in li.find_all("span"):
            txt = span.get_text(strip=True)
            if txt and 3 < len(txt) < 80 and ":" in txt:
                all_span_texts.append(txt)
    for span_text in all_span_texts:
        label_part = span_text.split(":", 1)[0].strip().lower()
        for key, patterns in LABEL_PATTERNS.items():
            if result[key] is not None:
                continue
            for pat in patterns:
                if label_part == pat:
                    result[key] = parse_colon_text(span_text, key)
                    break
    return result

def parse_next_data(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag  = soup.find("script", id="__NEXT_DATA__")
        if not tag or not tag.string:
            return None
        nd = json.loads(tag.string)

        def find_prop(obj, depth=0):
            if depth > 8: return None
            if isinstance(obj, dict):
                if any(k in obj for k in ["built_in_year","parking_spaces","servant_quarters"]):
                    return obj
                for v in obj.values():
                    r = find_prop(v, depth+1)
                    if r: return r
            elif isinstance(obj, list):
                for item in obj:
                    r = find_prop(item, depth+1)
                    if r: return r
            return None

        prop = find_prop(nd)
        if not prop: return None

        result = {
            "built_in_year":    safe_year(
                                    prop.get("built_in_year") or
                                    prop.get("year_built") or
                                    prop.get("construction_year")),
            "parking":          safe_count(
                                    prop.get("parking_spaces") or
                                    prop.get("car_parking") or
                                    prop.get("parking"), 10),
            "servant_quarters": safe_count(
                                    prop.get("servant_quarters") or
                                    prop.get("servants_quarters"), 5),
            "store_rooms":      safe_count(
                                    prop.get("store_rooms") or
                                    prop.get("store_room"), 5),
            "kitchens":         safe_count(
                                    prop.get("kitchens") or
                                    prop.get("kitchen"), 5),
            "drawing_rooms":    safe_count(
                                    prop.get("drawing_rooms") or
                                    prop.get("drawing_room"), 5),
        }
        if any(v is not None for v in result.values()):
            return result
    except:
        pass
    return None

def scrape_detail_page(url):
    if not url or url == "N/A":
        return url, dict(EMPTY_EXTRAS)

    time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))

    for attempt in range(2):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [429] sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                time.sleep(2 * (attempt + 1))
                continue

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            nd_result   = parse_next_data(html)
            html_result = parse_amenities(soup)

            merged = dict(EMPTY_EXTRAS)
            for k in merged:
                if nd_result and nd_result.get(k) is not None:
                    merged[k] = nd_result[k]
                elif html_result and html_result.get(k) is not None:
                    merged[k] = html_result[k]

            return url, merged

        except Exception as e:
            if attempt < 1:
                time.sleep(1)
            else:
                print(f"    Failed ({url[:55]}): {e}")

    return url, dict(EMPTY_EXTRAS)

# ── Parallel batch fetcher ────────────────────────────────────
def fetch_details_parallel(listings, scraped_urls):
    new_map = {
        l["url"]: l for l in listings
        if l.get("url","") not in scraped_urls and l.get("url","") != "N/A"
    }
    if not new_map:
        return 0

    enriched = 0
    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {pool.submit(scrape_detail_page, u): u for u in new_map}
        for future in as_completed(futures):
            url, extras = future.result()
            if url in new_map:
                new_map[url].update(extras)
                scraped_urls.add(url)
                enriched += 1
    return enriched


# ═══════════════════════════════════════════════════════════════
#  LIST PAGE CARD PARSER — House only
# ═══════════════════════════════════════════════════════════════

NOT_HOUSE_KEYWORDS = [
    "flat", "apartment", "penthouse", "upper portion", "lower portion",
    "farm house", "plot", "office", "shop", "room", "studio",
]

def parse_cards(soup):
    listings = []
    cards = soup.find_all("li", role="article")
    if not cards:
        cards = soup.find_all("article")
    print(f"  Found {len(cards)} cards  ", end="")

    skipped = 0
    for card in cards:
        try:
            price = None
            el = card.find(attrs={"aria-label": "Price"})
            if el:
                price = el.text.strip()
            else:
                for span in card.find_all("span"):
                    if span.text and any(x in span.text for x in ["Crore","Lakh","Arab","PKR"]):
                        price = span.text.strip()
                        break
            if not price:
                continue

            area = None
            el = card.find(attrs={"aria-label": "Area"})
            if el:
                area = el.text.strip()
            else:
                for span in card.find_all("span"):
                    if span.text and any(x in span.text for x in ["Marla","Kanal","Sq. Yd."]):
                        area = span.text.strip()
                        break

            el    = card.find(attrs={"aria-label": "Beds"})
            beds  = el.text.strip() if el else None
            el    = card.find(attrs={"aria-label": "Baths"})
            baths = el.text.strip() if el else None
            el    = card.find(attrs={"aria-label": "Location"})
            loc   = el.text.strip() if el else None

            el    = card.find(attrs={"aria-label": "Title"}) or card.find("h2")
            title = el.text.strip().lower() if el else ""

            if any(kw in title for kw in NOT_HOUSE_KEYWORDS):
                skipped += 1
                continue

            link = card.find("a", href=True)
            url  = link["href"] if link else "N/A"
            if url and url.startswith("/"):
                url = "https://www.zameen.com" + url

            listings.append({
                "price": price, "area": area, "city": "Islamabad",
                "bedrooms": beds, "bathrooms": baths,
                "location": loc, "property_type": "House", "url": url,
                "built_in_year": None, "parking": None,
                "servant_quarters": None, "store_rooms": None,
                "kitchens": None, "drawing_rooms": None,
            })

        except Exception as e:
            print(f"\n  Card error: {e}")

    print(f"→  kept {len(listings)}  skipped {skipped}")
    return listings


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def scrape():
    print("=" * 65)
    print("  Zameen.com Islamabad — HOUSE Scraper (Speed Optimised)")
    print(f"  Target  : {TARGET_COUNT:,} listings  ({MAX_PAGES} pages max)")
    print(f"  Threads : {DETAIL_WORKERS} parallel workers")
    print(f"  Timeout : {REQUEST_TIMEOUT}s per request")
    print("=" * 65 + "\n")

    driver   = get_driver()
    all_data = []

    if os.path.exists(OUTPUT_CSV):
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            existing = existing[existing["property_type"] == "House"]
            all_data = existing.to_dict("records")
            print(f"[RESUME] Loaded {len(all_data)} existing House listings\n")
        except:
            pass

    scraped_urls = {r.get("url","") for r in all_data}

    def fetch_list_page(drv, url):
        try:
            drv.get(url)
            time.sleep(random.uniform(PAGE_WAIT_MIN, PAGE_WAIT_MAX))
        except Exception as e:
            print(f"  Load issue: {e}")
        try:
            return BeautifulSoup(drv.page_source, "html.parser")
        except:
            return None

    for page in range(1, MAX_PAGES + 1):
        list_url = f"https://www.zameen.com/Homes/Islamabad-3-{page}.html"
        print(f"[Page {page:>3}/{MAX_PAGES}]  ", end="")

        soup     = fetch_list_page(driver, list_url)
        listings = parse_cards(soup) if soup else []

        if not listings:
            print("  0 houses — restarting driver, retry in 15s...")
            try: driver.quit()
            except: pass
            time.sleep(15)
            driver   = get_driver()
            soup     = fetch_list_page(driver, list_url)
            listings = parse_cards(soup) if soup else []
            if not listings:
                print("  Still 0 — skipping page")
                continue

        new_count = sum(1 for l in listings if l.get("url","") not in scraped_urls)
        print(f"  Fetching {new_count} new detail pages ({DETAIL_WORKERS} threads)...")

        t0       = time.time()
        enriched = fetch_details_parallel(listings, scraped_urls)
        elapsed  = time.time() - t0
        rate     = enriched / max(elapsed, 1)
        print(f"  ✓ {enriched} done in {elapsed:.1f}s  ({rate:.1f} pages/sec)")

        all_data.extend(listings)

        seen, deduped = set(), []
        for l in all_data:
            k = (l.get("price",""), l.get("area",""), l.get("location",""))
            if k not in seen:
                seen.add(k); deduped.append(l)
        all_data = deduped

        pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)

        df_tmp = pd.DataFrame(all_data)
        extras = ["built_in_year","parking","servant_quarters",
                  "store_rooms","kitchens","drawing_rooms"]
        fills  = {c: f"{df_tmp[c].notna().mean()*100:.0f}%"
                  for c in extras if c in df_tmp.columns}
        print(f"  Total: {len(all_data):,} houses  |  Fill: {fills}\n")

        if len(all_data) >= TARGET_COUNT:
            print(f"[TARGET MET] {len(all_data):,} listings collected!")
            break

        time.sleep(random.uniform(1.5, 3.0))

    driver.quit()
    pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 65)
    print(f"  DONE — {len(all_data):,} House listings → {OUTPUT_CSV}")
    print("=" * 65)
    print("\nNext: run train_model.ipynb")


if __name__ == "__main__":
    scrape()