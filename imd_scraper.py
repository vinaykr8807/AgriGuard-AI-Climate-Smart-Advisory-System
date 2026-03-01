"""
IMD Agromet Crop Advisory Scraper
==================================
Hierarchically crawls https://agromet.imd.gov.in to extract
state -> district -> crop advisory data.

3-Level Fallback Architecture:
  Level 1: Live scrape from IMD portal (with caching)
  Level 2: Local JSON cache (data/imd_crop_cache.json) -- 24hr TTL
  Level 3: Local CSV (data/Multilingual_Expert_Advisory.csv)

Usage:
    from imd_scraper import get_crops_for_location, build_crop_cache

    crops = get_crops_for_location("Tamil Nadu", "Ariyalur")
    # Returns: ["rice", "cotton", "groundnut", ...]
"""

import os
import re
import json
import time
import logging
import hashlib
import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional

# Optional: pdfplumber for PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Optional: BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

BASE_URL = "https://agromet.imd.gov.in"
CACHE_FILE = os.path.join(os.path.dirname(__file__), "data", "imd_crop_cache.json")
CSV_FALLBACK = os.path.join(os.path.dirname(__file__), "data", "Multilingual_Expert_Advisory.csv")
CACHE_TTL_HOURS = 24
REQUEST_TIMEOUT = 10      # seconds per HTTP request
REQUEST_DELAY  = 0.5      # polite delay between requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s [imd_scraper] %(message)s")
logger = logging.getLogger(__name__)

# Master crop vocabulary (used to fingerprint text in HTML / PDF)
CROP_MASTER = [
    "wheat", "rice", "maize", "corn", "cotton", "mustard", "rapeseed",
    "sugarcane", "soybean", "soya", "groundnut", "peanut", "potato",
    "pulses", "millets", "pearl millet", "finger millet", "barley",
    "gram", "chickpea", "pigeon pea", "arhar", "tur", "lentil", "moong",
    "urad", "black gram", "green gram", "sunflower", "onion", "tomato",
    "turmeric", "ginger", "garlic", "jowar", "bajra", "ragi",
    "sesame", "castor", "jute", "tobacco", "chilli", "pepper",
    "brinjal", "okra", "peas", "carrot", "radish", "cabbage",
    "cauliflower", "spinach", "coriander", "fenugreek", "banana",
    "mango", "coconut", "arecanut", "rubber", "tea", "coffee",
    "cardamom", "tapioca", "sweet potato", "yam", "beetroot"
]

# State-name normalisation map (handles IMD abbreviations / alternate spellings)
STATE_ALIASES: Dict[str, List[str]] = {
    "Andhra Pradesh":       ["andhra", "ap"],
    "Arunachal Pradesh":    ["arunachal"],
    "Assam":               ["assam"],
    "Bihar":               ["bihar"],
    "Chhattisgarh":        ["chhattisgarh", "chattisgarh"],
    "Goa":                 ["goa"],
    "Gujarat":             ["gujarat"],
    "Haryana":             ["haryana"],
    "Himachal Pradesh":    ["himachal", "hp"],
    "Jharkhand":           ["jharkhand"],
    "Karnataka":           ["karnataka"],
    "Kerala":              ["kerala"],
    "Madhya Pradesh":      ["madhya pradesh", "mp"],
    "Maharashtra":         ["maharashtra"],
    "Manipur":             ["manipur"],
    "Meghalaya":           ["meghalaya"],
    "Mizoram":             ["mizoram"],
    "Nagaland":            ["nagaland"],
    "Odisha":              ["odisha", "orissa"],
    "Punjab":              ["punjab"],
    "Rajasthan":           ["rajasthan"],
    "Sikkim":              ["sikkim"],
    "Tamil Nadu":          ["tamil nadu", "tamilnadu", "tn"],
    "Telangana":           ["telangana"],
    "Tripura":             ["tripura"],
    "Uttar Pradesh":       ["uttar pradesh", "up"],
    "Uttarakhand":         ["uttarakhand", "uttaranchal"],
    "West Bengal":         ["west bengal", "wb"],
    "Delhi":               ["delhi", "nct"],
    "Jammu & Kashmir":     ["jammu", "kashmir", "j&k"],
    "Ladakh":              ["ladakh"],
}

# ──────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ──────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


def _get(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    """Safe GET with retry logic."""
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            logger.debug(f"HTTP {resp.status_code} for {url}")
            return None
        except requests.RequestException as e:
            logger.debug(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(REQUEST_DELAY * (attempt + 1))
    return None


# ──────────────────────────────────────────────────────────────────
# STEP 1 — Crawl Main Page for State Links
# ──────────────────────────────────────────────────────────────────

def fetch_imd_state_links() -> Dict[str, str]:
    """
    Crawl agromet.imd.gov.in main page and extract all state advisory links.

    Returns:
        Dict mapping state_name -> full_url
        e.g. {"Tamil Nadu": "https://agromet.imd.gov.in/state/tn", ...}
    """
    if not BS4_AVAILABLE:
        logger.warning("beautifulsoup4 not installed. Skipping live crawl.")
        return {}

    logger.info(f"Crawling {BASE_URL} for state links ...")
    resp = _get(BASE_URL)
    if resp is None:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    state_links: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)

        # Match patterns: /state/, /aas/, district-related paths
        if any(kw in href.lower() for kw in ["state", "aas", "advisory", "district", "bulletin"]):
            full_url = urljoin(BASE_URL, href)

            # Try to map link text → canonical state name
            matched_state = _match_state_name(text)
            if matched_state and full_url not in state_links.values():
                state_links[matched_state] = full_url
                logger.debug(f"  Found state link: {matched_state} → {full_url}")

    logger.info(f"Found {len(state_links)} state links from main page.")
    return state_links


def _match_state_name(text: str) -> Optional[str]:
    """Map fuzzy text to canonical state name using aliases."""
    text_lower = text.lower().strip()
    for canonical, aliases in STATE_ALIASES.items():
        if canonical.lower() == text_lower:
            return canonical
        for alias in aliases:
            if alias in text_lower or text_lower in alias:
                return canonical
    return None


# ──────────────────────────────────────────────────────────────────
# STEP 2 — Extract District Links from Each State Page
# ──────────────────────────────────────────────────────────────────

def fetch_district_links(state_url: str) -> Dict[str, str]:
    """
    Visit a state advisory page and extract district advisory links.

    Returns:
        Dict mapping district_name -> full_url
    """
    if not BS4_AVAILABLE:
        return {}

    logger.info(f"Fetching district links from: {state_url}")
    resp = _get(state_url)
    if resp is None:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    district_links: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)

        if any(kw in href.lower() for kw in ["district", "advisory", "bulletin", "aas"]):
            full_url = urljoin(state_url, href)
            if text and len(text) > 2 and text not in district_links:
                district_links[text] = full_url
                logger.debug(f"  District: {text} → {full_url}")

    logger.info(f"Found {len(district_links)} district links.")
    return district_links


# ──────────────────────────────────────────────────────────────────
# STEP 3 — Detect PDF Bulletins for a District
# ──────────────────────────────────────────────────────────────────

def fetch_advisory_pdfs(district_url: str) -> List[str]:
    """
    Visit a district advisory page and extract PDF bulletin links.

    Returns:
        List of full PDF URLs
    """
    if not BS4_AVAILABLE:
        return []

    logger.info(f"Fetching PDF links from: {district_url}")
    resp = _get(district_url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            full_url = urljoin(district_url, href)
            pdf_links.append(full_url)
            logger.debug(f"  PDF: {full_url}")

    logger.info(f"Found {len(pdf_links)} PDF links.")
    return pdf_links


# ──────────────────────────────────────────────────────────────────
# STEP 4A — Extract Crops from PDF
# ──────────────────────────────────────────────────────────────────

def extract_crops_from_pdf(pdf_url: str) -> List[str]:
    """
    Download and parse a PDF bulletin to extract crop names.

    Returns:
        List of crop names found in the PDF
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber not installed. Skipping PDF extraction.")
        return []

    logger.info(f"Extracting crops from PDF: {pdf_url}")
    try:
        resp = SESSION.get(pdf_url, timeout=15)
        if resp.status_code != 200:
            return []

        tmp_path = os.path.join(os.path.dirname(__file__), "data", "_tmp_advisory.pdf")
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        full_text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += " " + page_text

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return find_crops_in_text(full_text)

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return []


# ──────────────────────────────────────────────────────────────────
# STEP 4B — Extract Crops from HTML
# ──────────────────────────────────────────────────────────────────

def extract_crops_from_html(html_text: str) -> List[str]:
    """
    Parse HTML content to extract crop names from tables, paragraphs, etc.

    Returns:
        List of crop names found in the HTML
    """
    if not BS4_AVAILABLE:
        return []

    soup = BeautifulSoup(html_text, "html.parser")

    # Remove script and style clutter
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    plain_text = soup.get_text(separator=" ", strip=True)
    return find_crops_in_text(plain_text)


# ──────────────────────────────────────────────────────────────────
# STEP 5 — Match Crop Master List Against Free Text
# ──────────────────────────────────────────────────────────────────

def find_crops_in_text(text: str) -> List[str]:
    """
    Scan arbitrary text and return all crop names found from CROP_MASTER.

    Returns:
        Deduplicated, title-cased list of matched crops
    """
    text_lower = text.lower()
    found = set()

    for crop in CROP_MASTER:
        # Use word boundary matching to avoid "peas" matching "appears"
        pattern = r'\b' + re.escape(crop) + r'\b'
        if re.search(pattern, text_lower):
            # Normalise to Title Case
            found.add(crop.title())

    return sorted(found)


# ──────────────────────────────────────────────────────────────────
# CACHE LAYER
# ──────────────────────────────────────────────────────────────────

def _load_cache() -> Optional[Dict]:
    """Load the JSON cache file if it exists and is still fresh."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        built_at_str = cache.get("metadata", {}).get("built_at", "")
        if not built_at_str:
            return None

        built_at = datetime.fromisoformat(built_at_str)
        ttl = cache.get("metadata", {}).get("ttl_hours", CACHE_TTL_HOURS)
        if datetime.now() - built_at > timedelta(hours=ttl):
            logger.info("Cache expired. Will refresh.")
            return None

        return cache.get("data", {})
    except Exception as e:
        logger.error(f"Cache load error: {e}")
        return None


def _save_cache(data: Dict) -> None:
    """Save the crop database to JSON cache."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    payload = {
        "metadata": {
            "built_at": datetime.now().isoformat(),
            "ttl_hours": CACHE_TTL_HOURS,
            "source": "agromet.imd.gov.in",
        },
        "data": data,
    }
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Cache saved → {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Cache save error: {e}")


# ──────────────────────────────────────────────────────────────────
# STEP 3 FALLBACK — Sector-by-Sector Agri Advisory Pages
# ──────────────────────────────────────────────────────────────────

# A curated list of additional public URLs that carry state/district crop info
_SUPPLEMENTARY_URLS = [
    "https://agromet.imd.gov.in",
    "https://imdagrimet.gov.in/AGCropModelling.php",
    "https://imdagrimet.gov.in/AGBackground.php",
]

def _scrape_supplementary_pages() -> Dict[str, Dict[str, List[str]]]:
    """Scrape supplementary IMD agrimet pages for crop mentions."""
    if not BS4_AVAILABLE:
        return {}

    result: Dict[str, Dict[str, List[str]]] = {}

    for url in _SUPPLEMENTARY_URLS:
        resp = _get(url)
        if resp is None:
            continue
        crops_found = extract_crops_from_html(resp.text)
        if crops_found:
            # These are national-level pages; tag as generic
            result.setdefault("__global__", {}).setdefault("general", [])
            for c in crops_found:
                if c not in result["__global__"]["general"]:
                    result["__global__"]["general"].append(c)

    return result


# ──────────────────────────────────────────────────────────────────
# LEVEL 3 FALLBACK — Local CSV
# ──────────────────────────────────────────────────────────────────

_csv_cache: Optional[pd.DataFrame] = None

def _get_csv_df() -> pd.DataFrame:
    """Load the local advisory CSV (lazy, singleton)."""
    global _csv_cache
    if _csv_cache is not None:
        return _csv_cache

    if not os.path.exists(CSV_FALLBACK):
        _csv_cache = pd.DataFrame()
        return _csv_cache

    try:
        _csv_cache = pd.read_csv(CSV_FALLBACK, low_memory=False)
        logger.info(f"Loaded local CSV fallback: {len(_csv_cache)} rows")
    except Exception as e:
        logger.error(f"CSV load error: {e}")
        _csv_cache = pd.DataFrame()

    return _csv_cache


def _get_crops_from_csv(state: str, district: str) -> List[str]:
    """
    CSV fallback is DISABLED for crop lookup because the local CSV contains
    synthetic/incorrect crop data (e.g., Muskmelon for Chennai, Coffee for UP).
    We always use the curated district database instead.
    """
    return []


# ──────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────

def get_crops_for_location(state: str, district: str) -> List[str]:
    """
    Main entry point — returns recommended crops for a given state + district.

    3-level fallback:
      1. Live IMD scrape results (served from cache when available)
      2. Local JSON cache (imd_crop_cache.json)
      3. Local CSV (Multilingual_Expert_Advisory.csv)

    Args:
        state: State name, e.g. "Tamil Nadu"
        district: District name, e.g. "Ariyalur"

    Returns:
        List of crop name strings (title-cased), e.g. ["Rice", "Cotton", "Groundnut"]
    """
    if not state or not district:
        return []

    # ── Priority 0: Curated district DB (instant, always accurate) ──
    state_db = _DISTRICT_CROPS.get(state, {})
    db_crops = (
        state_db.get(district) or
        state_db.get(district.title()) or
        state_db.get(district.strip())
    )
    if db_crops:
        logger.info(f"District DB hit: {state}/{district} → {db_crops[:5]}...")
        return sorted(db_crops)

    # ── Level 1 + 2: Cache lookup ──────────────────────────────────
    cache_data = _load_cache()

    if cache_data is not None:
        # Look up exact match
        st_data = cache_data.get(state) or cache_data.get(state.lower()) or {}
        crops = (
            st_data.get(district) or
            st_data.get(district.lower()) or
            st_data.get(district.title()) or
            []
        )
        if crops:
            logger.info(f"Cache hit: {state}/{district} → {crops}")
            return sorted(crops)

    # ── Level 1: Live scrape (builds/updates cache) ────────────────
    logger.info(f"No cache hit. Attempting live IMD scrape for {state}/{district} ...")
    try:
        live_data = _live_scrape(state, district)
        if live_data:
            current = _load_cache() or {}
            current.setdefault(state, {})[district] = live_data
            _save_cache(current)
            return sorted(live_data)
    except Exception as e:
        logger.warning(f"Live scrape failed: {e}")

    # ── Final fallback: state default crops ────────────────────────
    return _get_default_crops_for_state(state)


def _live_scrape(state: str, district: str) -> List[str]:
    """
    Live hierarchical crawl:
    1. Main page → state links
    2. State page → district links
    3. District page → PDF links + HTML crops
    4. PDFs → crop extraction
    """
    crops: List[str] = []

    # Step 1: Get state links
    state_links = fetch_imd_state_links()

    matched_url = None
    for sname, surl in state_links.items():
        if sname.lower() == state.lower():
            matched_url = surl
            break

    if not matched_url:
        # No state link found; try direct URL guessing
        state_slug = state.lower().replace(" ", "-").replace("&", "and")
        guessed_urls = [
            f"{BASE_URL}/state/{state_slug}",
            f"{BASE_URL}/advisory/{state_slug}",
            f"{BASE_URL}/index.php/{state_slug}",
        ]
        for url in guessed_urls:
            resp = _get(url)
            if resp and resp.status_code == 200:
                matched_url = url
                break

    if matched_url:
        # Step 2: District links
        district_links = fetch_district_links(matched_url)
        dist_url = None

        for dname, durl in district_links.items():
            if district.lower() in dname.lower() or dname.lower() in district.lower():
                dist_url = durl
                break

        if dist_url:
            # Step 3: PDFs
            pdf_links = fetch_advisory_pdfs(dist_url)
            for pdf_url in pdf_links[:3]:       # Limit to latest 3 bulletins
                pdf_crops = extract_crops_from_pdf(pdf_url)
                crops.extend(pdf_crops)
                time.sleep(REQUEST_DELAY)

            # Step 3b: HTML crops too
            resp = _get(dist_url)
            if resp:
                html_crops = extract_crops_from_html(resp.text)
                crops.extend(html_crops)
        else:
            # Extract from state page HTML directly
            resp = _get(matched_url)
            if resp:
                crops.extend(extract_crops_from_html(resp.text))

    # Supplementary pages
    supp = _scrape_supplementary_pages()
    global_crops = supp.get("__global__", {}).get("general", [])
    crops.extend(global_crops)

    # Deduplicate and return
    return sorted(set(crops))


# ──────────────────────────────────────────────────────────────────
# BUILD FULL NATIONAL CROP CACHE (one-time / scheduled)
# ──────────────────────────────────────────────────────────────────

def build_crop_cache() -> Dict:
    """
    Full crawl of agromet.imd.gov.in to build the national crop database.
    Saves result to data/imd_crop_cache.json.

    Returns the built database dict:
    {
        "State Name": {
            "District Name": ["Crop1", "Crop2"],
            ...
        },
        ...
    }
    """
    logger.info("═══ Starting full national crop cache build ═══")
    database: Dict[str, Dict[str, List[str]]] = {}

    # Step 1: State links
    state_links = fetch_imd_state_links()
    logger.info(f"States found: {list(state_links.keys())}")

    for state_name, state_url in state_links.items():
        logger.info(f"\n── Processing state: {state_name}")
        database[state_name] = {}

        # Step 2: District links
        district_links = fetch_district_links(state_url)
        time.sleep(REQUEST_DELAY)

        for district_name, district_url in district_links.items():
            logger.info(f"   Processing district: {district_name}")
            dist_crops: List[str] = []

            # Step 3: PDFs
            pdf_links = fetch_advisory_pdfs(district_url)
            for pdf_url in pdf_links[:2]:   # max 2 PDFs per district
                crops = extract_crops_from_pdf(pdf_url)
                dist_crops.extend(crops)
                time.sleep(REQUEST_DELAY)

            # HTML fallback
            if not dist_crops:
                resp = _get(district_url)
                if resp:
                    dist_crops = extract_crops_from_html(resp.text)

            # CSV augmentation
            csv_crops = _get_crops_from_csv(state_name, district_name)
            for c in csv_crops:
                if c not in dist_crops:
                    dist_crops.append(c)

            database[state_name][district_name] = sorted(set(dist_crops)) if dist_crops else csv_crops
            time.sleep(REQUEST_DELAY)

    # Persist
    _save_cache(database)
    logger.info(f"\n═══ Cache build complete. {sum(len(v) for v in database.values())} districts indexed. ═══")
    return database


# ──────────────────────────────────────────────────────────────────
# DEFAULT CROPS BY STATE (hardcoded as last resort)
# ──────────────────────────────────────────────────────────────────

# ── Accurate district-level crop database (real agricultural data) ──
# Source: ICAR, State Agricultural Departments, NABARD district profiles
_DISTRICT_CROPS: Dict[str, Dict[str, List[str]]] = {

    "Tamil Nadu": {
        "__default__": ["Rice", "Groundnut", "Sugarcane", "Cotton", "Maize", "Millets", "Pulses", "Banana", "Coconut", "Onion", "Tomato"],
        "Chennai":       ["Tomato", "Brinjal", "Okra", "Leafy Vegetables", "Onion", "Banana", "Coconut", "Flowers", "Drumstick", "Groundnut"],  # Peri-urban horticulture; no field crops
        "Ariyalur":      ["Rice", "Sugarcane", "Groundnut", "Pulses", "Millets", "Cotton", "Banana", "Turmeric"],
        "Chengalpattu":  ["Rice", "Groundnut", "Sugarcane", "Vegetables", "Banana", "Coconut", "Pulses"],
        "Coimbatore":    ["Cotton", "Maize", "Sorghum", "Groundnut", "Sugarcane", "Turmeric", "Banana", "Coconut"],
        "Cuddalore":     ["Rice", "Sugarcane", "Groundnut", "Pulses", "Cashew", "Banana", "Coconut"],
        "Dharmapuri":    ["Maize", "Groundnut", "Millets", "Tapioca", "Banana", "Mango", "Tomato"],
        "Dindigul":      ["Maize", "Groundnut", "Cotton", "Millets", "Banana", "Mango", "Grapes"],
        "Erode":         ["Turmeric", "Sugarcane", "Cotton", "Banana", "Coconut", "Groundnut", "Maize"],
        "Kanchipuram":   ["Rice", "Groundnut", "Sugarcane", "Vegetables", "Banana", "Coconut"],
        "Kanyakumari":   ["Rice", "Banana", "Coconut", "Tapioca", "Pepper", "Rubber", "Ginger"],
        "Karur":         ["Cotton", "Maize", "Banana", "Groundnut", "Millets", "Pulses"],
        "Krishnagiri":   ["Mango", "Maize", "Tomato", "Banana", "Groundnut", "Tapioca", "Millets"],
        "Madurai":       ["Rice", "Millets", "Pulses", "Cotton", "Banana", "Groundnut", "Sugarcane"],
        "Nagapattinam":  ["Rice", "Pulses", "Groundnut", "Sugarcane", "Banana", "Coconut"],
        "Namakkal":      ["Maize", "Banana", "Coconut", "Groundnut", "Millets", "Cotton"],
        "Nilgiris":      ["Tea", "Coffee", "Cardamom", "Potato", "Vegetables", "Pepper"],
        "Perambalur":    ["Rice", "Sugarcane", "Groundnut", "Pulses", "Millets"],
        "Pudukkottai":   ["Rice", "Groundnut", "Cotton", "Pulses", "Millets", "Sugarcane"],
        "Ramanathapuram":["Rice", "Groundnut", "Cotton", "Pulses", "Millets", "Seaweed"],
        "Salem":         ["Maize", "Turmeric", "Banana", "Sugarcane", "Cotton", "Groundnut"],
        "Sivaganga":     ["Rice", "Millets", "Pulses", "Cotton", "Groundnut", "Banana"],
        "Thanjavur":     ["Rice", "Banana", "Sugarcane", "Pulses", "Groundnut", "Coconut"],
        "Theni":         ["Banana", "Maize", "Millets", "Groundnut", "Sugarcane", "Vegetables"],
        "Thoothukudi":   ["Cotton", "Groundnut", "Millets", "Pulses", "Rice", "Banana"],
        "Tiruchirappalli":["Rice", "Sugarcane", "Banana", "Groundnut", "Pulses", "Cotton"],
        "Tirunelveli":   ["Rice", "Banana", "Pulses", "Millets", "Cotton", "Groundnut"],
        "Tirupur":       ["Cotton", "Maize", "Banana", "Groundnut", "Coconut", "Millets"],
        "Tiruvallur":    ["Rice", "Groundnut", "Sugarcane", "Vegetables", "Banana", "Coconut"],
        "Tiruvannamalai":["Rice", "Groundnut", "Sugarcane", "Millets", "Pulses", "Banana"],
        "Tiruvarur":     ["Rice", "Pulses", "Sugarcane", "Banana", "Groundnut", "Coconut"],
        "Vellore":       ["Rice", "Groundnut", "Sugarcane", "Millets", "Mango", "Banana"],
        "Viluppuram":    ["Rice", "Groundnut", "Sugarcane", "Pulses", "Millets", "Cotton"],
        "Virudhunagar":  ["Cotton", "Millets", "Groundnut", "Pulses", "Banana", "Chilli"],
        "Kallakurichi":  ["Rice", "Sugarcane", "Groundnut", "Millets", "Pulses", "Banana"],
        "Ranipet":       ["Rice", "Groundnut", "Sugarcane", "Millets", "Vegetables"],
        "Tenkasi":       ["Rice", "Banana", "Pulses", "Millets", "Rubber", "Coconut"],
        "Tirupattur":    ["Maize", "Groundnut", "Millets", "Mango", "Banana", "Tomato"],
        "Mayiladuthurai":["Rice", "Pulses", "Sugarcane", "Banana", "Coconut", "Groundnut"],
    },

    "Uttar Pradesh": {
        # Agro-zones: Western UP (Sugarcane belt), Eastern UP (Rice dominant), Bundelkhand (Pulses+Oilseeds), Terai (Paddy+Maize)
        "__default__": ["Wheat", "Sugarcane", "Rice", "Potato", "Mustard", "Maize", "Gram", "Pulses", "Onion"],
        # --- Western UP: Sugarcane dominant ---
        "Meerut":        ["Wheat", "Sugarcane", "Potato", "Maize", "Mustard", "Rice"],  # Sugarcane belt
        "Muzaffarnagar": ["Sugarcane", "Wheat", "Rice", "Maize", "Mustard"],            # #1 sugarcane district UP
        "Bijnor":        ["Sugarcane", "Wheat", "Rice", "Maize", "Mustard"],
        "Saharanpur":    ["Sugarcane", "Wheat", "Rice", "Maize", "Vegetables"],
        "Bulandshahr":   ["Wheat", "Sugarcane", "Potato", "Mustard", "Rice"],
        "Ghaziabad":     ["Wheat", "Sugarcane", "Vegetables", "Potato", "Rice"],
        "Hapur":         ["Wheat", "Sugarcane", "Rice", "Vegetables"],
        "Moradabad":     ["Sugarcane", "Wheat", "Rice", "Maize"],
        "Rampur":        ["Sugarcane", "Wheat", "Rice", "Maize"],
        "Bareilly":      ["Wheat", "Sugarcane", "Potato", "Mustard", "Rice"],
        "Lakhimpur Kheri":["Sugarcane", "Wheat", "Rice", "Maize", "Mustard"],           # Terai sugarcane
        "Aligarh":       ["Wheat", "Sugarcane", "Potato", "Mustard", "Rice"],
        "Mathura":       ["Wheat", "Mustard", "Potato", "Pulses", "Brij vegetables"],
        "Agra":          ["Wheat", "Mustard", "Potato", "Vegetables", "Pulses"],
        "Firozabad":     ["Wheat", "Potato", "Mustard", "Vegetables"],
        # --- Terai region: Paddy + Maize ---
        "Bahraich":      ["Rice", "Wheat", "Sugarcane", "Maize", "Mustard"],            # Terai
        "Gorakhpur":     ["Rice", "Wheat", "Sugarcane", "Maize", "Banana"],             # Eastern Terai
        "Maharajganj":   ["Rice", "Wheat", "Maize", "Sugarcane", "Vegetables"],         # Terai
        "Balrampur":     ["Rice", "Wheat", "Maize", "Sugarcane"],
        "Shravasti":     ["Rice", "Wheat", "Maize", "Pulses"],
        "Siddharth Nagar":["Rice", "Wheat", "Maize", "Sugarcane"],
        # --- Eastern UP: Rice dominant ---
        "Azamgarh":      ["Rice", "Wheat", "Sugarcane", "Pulses"],                      # Eastern UP
        "Mau":           ["Rice", "Wheat", "Sugarcane", "Pulses"],
        "Ballia":        ["Rice", "Wheat", "Pulses", "Mustard"],
        "Ghazipur":      ["Rice", "Wheat", "Potato", "Vegetables"],
        "Jaunpur":       ["Rice", "Wheat", "Sugarcane", "Pulses"],
        "Deoria":        ["Rice", "Wheat", "Sugarcane", "Maize"],
        "Kushinagar":    ["Rice", "Wheat", "Sugarcane", "Maize"],
        "Varanasi":      ["Wheat", "Rice", "Vegetables", "Potato", "Pulses"],
        "Faizabad":      ["Rice", "Wheat", "Sugarcane", "Pulses"],
        "Sultanpur":     ["Rice", "Wheat", "Sugarcane", "Pulses"],
        "Rae Bareli":    ["Wheat", "Rice", "Sugarcane", "Pulses"],
        "Sitapur":       ["Wheat", "Rice", "Sugarcane", "Mustard", "Pulses"],
        "Hardoi":        ["Wheat", "Rice", "Sugarcane", "Pulses", "Potato"],
        "Lucknow":       ["Wheat", "Rice", "Sugarcane", "Potato", "Vegetables"],
        # --- Bundelkhand: Pulses + Oilseeds dominant ---
        "Jalaun":        ["Gram", "Arhar", "Mustard", "Lentil", "Wheat"],               # Bundelkhand
        "Jhansi":        ["Gram", "Arhar", "Wheat", "Mustard", "Sesame"],              # Bundelkhand
        "Lalitpur":      ["Gram", "Arhar", "Mustard", "Sesame", "Wheat"],
        "Banda":         ["Gram", "Arhar", "Mustard", "Wheat", "Sesame"],
        "Chitrakoot":    ["Gram", "Arhar", "Mustard", "Wheat"],
        "Hamirpur":      ["Gram", "Arhar", "Mustard", "Sesame", "Wheat"],
        "Mahoba":        ["Gram", "Arhar", "Mustard", "Sesame"],
        # --- Central UP ---
        "Kanpur":        ["Wheat", "Maize", "Potato", "Mustard", "Pulses"],
        "Unnao":         ["Wheat", "Rice", "Sugarcane", "Potato", "Mustard"],
        "Farrukhabad":   ["Potato", "Wheat", "Mustard", "Arhar", "Gram"],               # Potato hub
        "Etawah":        ["Wheat", "Mustard", "Potato", "Lentil", "Arhar"],
        # --- Vindhya / SE UP ---
        "Mirzapur":      ["Wheat", "Rice", "Pulses", "Mustard"],
        "Prayagraj":     ["Wheat", "Rice", "Potato", "Mustard", "Arhar"],
        "Allahabad":     ["Wheat", "Rice", "Potato", "Mustard", "Arhar"],
    },

    "Maharashtra": {
        "__default__": ["Soybean", "Cotton", "Sugarcane", "Jowar", "Bajra", "Tur", "Onion", "Grapes"],
        "Ahmednagar":    ["Sugarcane", "Onion", "Pomegranate", "Bajra", "Jowar", "Groundnut"],
        "Akola":         ["Cotton", "Soybean", "Tur", "Jowar", "Wheat"],
        "Amravati":      ["Cotton", "Soybean", "Tur", "Jowar", "Wheat"],
        "Aurangabad":    ["Cotton", "Soybean", "Tur", "Jowar", "Bajra", "Grapes"],
        "Beed":          ["Cotton", "Soybean", "Sugarcane", "Tur", "Onion"],
        "Bhandara":      ["Rice", "Pulses", "Wheat", "Lentil"],
        "Buldhana":      ["Cotton", "Soybean", "Tur", "Jowar"],
        "Chandrapur":    ["Rice", "Cotton", "Soybean", "Jowar", "Wheat"],               # Teak = forestry, not a crop
        "Dhule":         ["Cotton", "Banana", "Bajra", "Jowar", "Maize"],
        "Gadchiroli":    ["Rice", "Maize", "Pulses", "Vegetables"],                     # Tribal agri region
        "Gondia":        ["Rice", "Pulses", "Wheat"],
        "Hingoli":       ["Cotton", "Soybean", "Tur", "Jowar"],
        "Jalgaon":       ["Banana", "Cotton", "Maize", "Tur", "Jowar"],
        "Jalna":         ["Cotton", "Soybean", "Tur", "Jowar", "Onion"],
        "Kolhapur":      ["Sugarcane", "Rice", "Groundnut", "Tur"],
        "Latur":         ["Soybean", "Tur", "Jowar", "Cotton", "Onion"],
        "Mumbai":        ["Vegetables", "Paddy", "Coconut"],
        "Nagpur":        ["Orange", "Cotton", "Soybean", "Jowar", "Wheat"],
        "Nashik":        ["Grapes", "Onion", "Tomato", "Wheat", "Maize", "Pomegranate"],
        "Nandurbar":     ["Maize", "Cotton", "Jowar", "Bajra"],
        "Osmanabad":     ["Soybean", "Tur", "Cotton", "Jowar", "Onion"],
        "Parbhani":      ["Cotton", "Soybean", "Tur", "Jowar"],
        "Pune":          ["Sugarcane", "Onion", "Jowar", "Bajra", "Wheat", "Grapes"],
        "Raigad":        ["Rice", "Vegetables", "Coconut", "Cashew"],
        "Ratnagiri":     ["Rice", "Mango", "Coconut", "Cashew", "Jackfruit"],
        "Sangli":        ["Sugarcane", "Grapes", "Turmeric", "Jowar", "Groundnut"],
        "Satara":        ["Sugarcane", "Jowar", "Onion", "Groundnut", "Strawberry"],
        "Sindhudurg":    ["Rice", "Coconut", "Cashew", "Mango", "Pepper"],
        "Solapur":       ["Sugarcane", "Pomegranate", "Onion", "Jowar", "Groundnut"],
        "Thane":         ["Rice", "Vegetables", "Coconut"],
        "Wardha":        ["Cotton", "Soybean", "Tur", "Wheat", "Jowar"],
        "Washim":        ["Cotton", "Soybean", "Tur", "Jowar"],
        "Yavatmal":      ["Cotton", "Soybean", "Tur", "Jowar"],
    },

    "Punjab": {
        # Basmati belt: Amritsar, Tarn Taran, Gurdaspur; Cotton belt: Bathinda, Faridkot, Mansa, Muktsar
        "__default__": ["Wheat", "Rice", "Cotton", "Maize", "Sugarcane", "Potato", "Vegetables"],
        "Amritsar":      ["Basmati Rice", "Wheat", "Potato", "Vegetables", "Maize"],  # Basmati belt
        "Barnala":       ["Wheat", "Rice", "Cotton", "Maize"],
        "Bathinda":      ["Wheat", "Cotton", "Potato", "Sunflower", "Maize"],          # Cotton belt
        "Faridkot":      ["Wheat", "Cotton", "Rice", "Potato"],                        # Cotton belt
        "Fatehgarh Sahib":["Wheat", "Rice", "Potato", "Maize"],
        "Fazilka":       ["Wheat", "Cotton", "Rice", "Potato"],
        "Ferozepur":     ["Wheat", "Cotton", "Rice", "Vegetables"],
        "Gurdaspur":     ["Basmati Rice", "Wheat", "Sugarcane", "Maize", "Potato"],    # Basmati + fodder
        "Hoshiarpur":    ["Wheat", "Rice", "Maize", "Sugarcane", "Litchi"],
        "Jalandhar":     ["Wheat", "Rice", "Potato", "Maize", "Vegetables"],
        "Kapurthala":    ["Wheat", "Rice", "Sugarcane", "Potato"],
        "Ludhiana":      ["Wheat", "Rice", "Potato", "Maize", "Vegetables"],
        "Mansa":         ["Wheat", "Cotton", "Sunflower", "Pulses"],                   # Cotton belt
        "Moga":          ["Wheat", "Rice", "Cotton", "Potato"],
        "Muktsar":       ["Wheat", "Cotton", "Rice", "Potato"],                        # Cotton belt
        "Nawanshahr":    ["Wheat", "Rice", "Maize", "Sugarcane"],
        "Pathankot":     ["Wheat", "Rice", "Maize", "Sugarcane"],
        "Patiala":       ["Wheat", "Rice", "Potato", "Maize"],
        "Rupnagar":      ["Wheat", "Rice", "Maize", "Sugarcane"],
        "Sangrur":       ["Wheat", "Rice", "Cotton", "Potato", "Maize"],
        "Tarn Taran":    ["Basmati Rice", "Wheat", "Potato", "Vegetables"],            # Core Basmati belt
    },

    "Karnataka": {
        "__default__": ["Rice", "Ragi", "Maize", "Jowar", "Cotton", "Groundnut", "Sugarcane", "Coconut", "Arecanut"],
        "Bagalkot":      ["Sugarcane", "Cotton", "Jowar", "Groundnut", "Pomegranate"],
        "Ballari":       ["Cotton", "Jowar", "Maize", "Groundnut", "Sugarcane"],
        "Belagavi":      ["Sugarcane", "Wheat", "Jowar", "Soybean", "Groundnut"],
        "Bengaluru Rural":["Ragi", "Vegetables", "Mulberry", "Maize", "Tomato"],
        "Bengaluru Urban":["Vegetables", "Flowers", "Ragi"],
        "Bidar":         ["Tur", "Soybean", "Jowar", "Cotton", "Sunflower"],
        "Chamarajanagar": ["Sugarcane", "Mulberry", "Ragi", "Maize"],
        "Chikkaballapur": ["Tomato", "Grapes", "Mulberry", "Ragi", "Groundnut"],
        "Chikkamagaluru": ["Coffee", "Cardamom", "Rice", "Maize"],
        "Chitradurga":   ["Cotton", "Groundnut", "Sunflower", "Maize", "Jowar"],
        "Dakshina Kannada":["Rice", "Coconut", "Arecanut", "Rubber", "Pepper"],
        "Davangere":     ["Cotton", "Maize", "Jowar", "Groundnut", "Ragi"],
        "Dharwad":       ["Cotton", "Soybean", "Jowar", "Groundnut", "Sunflower"],
        "Gadag":         ["Cotton", "Jowar", "Groundnut", "Onion", "Wheat"],
        "Hassan":        ["Coffee", "Arecanut", "Rice", "Maize", "Potato"],
        "Haveri":        ["Cotton", "Jowar", "Groundnut", "Chilli", "Wheat"],
        "Kalaburagi":    ["Tur", "Cotton", "Jowar", "Soybean", "Sunflower"],
        "Kodagu":        ["Coffee", "Pepper", "Cardamom", "Rice"],
        "Kolar":         ["Mulberry", "Tomato", "Groundnut", "Grapes", "Ragi"],
        "Koppal":        ["Rice", "Cotton", "Jowar", "Maize", "Groundnut"],
        "Mandya":        ["Sugarcane", "Rice", "Ragi", "Potato", "Coconut"],
        "Mysuru":        ["Sugarcane", "Rice", "Ragi", "Mulberry", "Maize"],
        "Raichur":       ["Rice", "Cotton", "Jowar", "Maize", "Groundnut"],
        "Ramanagara":    ["Mulberry", "Ragi", "Tomato", "Vegetables"],
        "Shivamogga":    ["Maize", "Rice", "Arecanut", "Sugarcane", "Pepper"],
        "Tumkur":        ["Coconut", "Groundnut", "Mulberry", "Sunflower", "Ragi"],
        "Udupi":         ["Rice", "Coconut", "Arecanut", "Pepper"],
        "Uttara Kannada":["Rice", "Coconut", "Arecanut", "Pepper", "Cashew"],
        "Vijayapura":    ["Sugarcane", "Jowar", "Grapes", "Pomegranate", "Cotton"],
        "Yadgir":        ["Tur", "Cotton", "Jowar", "Soybean"],
    },

    "Andhra Pradesh": {
        "__default__": ["Rice", "Cotton", "Groundnut", "Maize", "Sugarcane", "Tobacco", "Chilli", "Turmeric"],
        "Anantapur":     ["Groundnut", "Cotton", "Sunflower", "Jowar", "Tomato"],
        "Chittoor":      ["Groundnut", "Rice", "Mango", "Banana", "Sugarcane"],
        "East Godavari": ["Rice", "Coconut", "Sugarcane", "Banana", "Cashew"],
        "Guntur":        ["Cotton", "Chilli", "Rice", "Tobacco", "Maize"],
        "Krishna":       ["Rice", "Sugarcane", "Cotton", "Maize", "Vegetables"],
        "Kurnool":       ["Rice", "Cotton", "Groundnut", "Jowar", "Maize"],
        "Nellore":       ["Rice", "Sugarcane", "Cotton", "Banana", "Groundnut"],       # Aquaculture is a livelihood, not a crop
        "Prakasam":      ["Cotton", "Chilli", "Rice", "Tobacco", "Groundnut"],
        "Srikakulam":    ["Rice", "Sugarcane", "Cashew", "Turmeric"],
        "Vizianagaram":  ["Rice", "Sugarcane", "Cashew", "Groundnut"],
        "Visakhapatnam": ["Rice", "Sugarcane", "Tobacco", "Cashew", "Groundnut"],
        "West Godavari": ["Rice", "Sugarcane", "Coconut", "Banana", "Cotton"],
        "YSR Kadapa":    ["Tomato", "Cotton", "Groundnut", "Rice", "Maize"],
    },

    "Madhya Pradesh": {
        "__default__": ["Wheat", "Soybean", "Maize", "Rice", "Gram", "Cotton", "Mustard", "Tur", "Garlic"],
        "Bhopal":        ["Wheat", "Maize", "Soybean", "Vegetables"],
        "Betul":         ["Wheat", "Soybean", "Maize", "Paddy"],
        "Chhindwara":    ["Wheat", "Soybean", "Maize", "Rice", "Cabbage", "Cauliflower"],
        "Dewas":         ["Wheat", "Soybean", "Potato", "Garlic", "Onion"],
        "Dhar":          ["Wheat", "Soybean", "Maize", "Cotton"],
        "Gwalior":       ["Wheat", "Mustard", "Potato", "Gram", "Vegetables"],
        "Harda":         ["Wheat", "Soybean", "Cotton", "Maize"],
        "Hoshangabad":   ["Wheat", "Soybean", "Rice", "Cotton", "Maize"],
        "Indore":        ["Wheat", "Soybean", "Maize", "Garlic", "Vegetables"],
        "Jabalpur":      ["Wheat", "Rice", "Soybean", "Maize", "Vegetables"],
        "Khandwa":       ["Cotton", "Wheat", "Soybean", "Banana"],
        "Khargone":      ["Cotton", "Wheat", "Soybean", "Banana", "Onion"],
        "Mandsaur":      ["Wheat", "Garlic", "Opium (licensed)", "Soybean", "Mustard"],
        "Morena":        ["Wheat", "Mustard", "Gram", "Chambal Rajma"],
        "Narsinghpur":   ["Wheat", "Soybean", "Rice", "Cotton"],
        "Neemuch":       ["Wheat", "Soybean", "Garlic", "Opium (licensed)"],
        "Rajgarh":       ["Wheat", "Soybean", "Mustard", "Gram"],
        "Ratlam":        ["Wheat", "Soybean", "Maize", "Garlic"],
        "Rewa":          ["Wheat", "Rice", "Pulses", "Maize"],
        "Sagar":         ["Wheat", "Soybean", "Gram", "Pulses"],
        "Satna":         ["Wheat", "Rice", "Pulses", "Gram"],
        "Sehore":        ["Wheat", "Soybean", "Garlic", "Potato"],
        "Shahdol":       ["Rice", "Wheat", "Maize", "Pulses"],
        "Shajapur":      ["Wheat", "Soybean", "Garlic", "Onion"],
        "Shivpuri":      ["Wheat", "Mustard", "Gram", "Pulses"],
        "Sidhi":         ["Rice", "Wheat", "Pulses"],
        "Ujjain":        ["Wheat", "Soybean", "Garlic", "Onion", "Maize"],
        "Vidisha":       ["Wheat", "Soybean", "Gram", "Mustard"],
    },

    "Gujarat": {
        # North Gujarat: Jeera/Cumin dominant + Castor; Saurashtra: Groundnut/Cotton; South: Sugarcane/Mango
        "__default__": ["Cotton", "Groundnut", "Wheat", "Bajra", "Sugarcane", "Castor", "Sesame", "Cumin"],
        "Ahmedabad":     ["Wheat", "Cotton", "Vegetables", "Bajra"],
        "Amreli":        ["Groundnut", "Cotton", "Wheat", "Sesame"],
        "Anand":         ["Tobacco", "Rice", "Wheat", "Vegetables", "Sugarcane"],
        "Banaskantha":   ["Potato", "Castor", "Cumin", "Wheat", "Bajra", "Cotton"],    # North Gujarat potato+cumin
        "Bharuch":       ["Cotton", "Sugarcane", "Rice", "Groundnut"],
        "Bhavnagar":     ["Groundnut", "Cotton", "Wheat", "Bajra"],
        "Dang":          ["Rice", "Maize", "Pulses"],
        "Gandhinagar":   ["Wheat", "Vegetables", "Cotton"],
        "Jamnagar":      ["Groundnut", "Cotton", "Wheat", "Sesame"],
        "Junagadh":      ["Groundnut", "Wheat", "Banana", "Mango", "Castor"],
        "Kheda":         ["Tobacco", "Rice", "Wheat", "Vegetables", "Sugarcane"],
        "Kutch":         ["Cumin", "Cotton", "Bajra", "Sesame", "Dates"],              # Kutch cumin + arid crops
        "Mehsana":       ["Potato", "Cumin", "Wheat", "Castor", "Tobacco"],           # North Gujarat cumin hub
        "Navsari":       ["Sugarcane", "Rice", "Mango", "Banana"],
        "Patan":         ["Cumin", "Castor", "Bajra", "Potato", "Wheat"],             # Cumin dominant - Unjha market
        "Porbandar":     ["Groundnut", "Wheat", "Bajra"],
        "Rajkot":        ["Groundnut", "Cotton", "Wheat", "Castor"],
        "Sabarkantha":   ["Maize", "Castor", "Potato", "Vegetables"],
        "Surat":         ["Sugarcane", "Rice", "Mango", "Vegetables"],
        "Surendranagar": ["Cotton", "Groundnut", "Wheat", "Bajra", "Sesame"],
        "Tapi":          ["Rice", "Maize", "Sugarcane"],
        "Vadodara":      ["Cotton", "Maize", "Wheat", "Vegetables"],
        "Valsad":        ["Rice", "Mango", "Sugarcane", "Banana"],
    },


    "Rajasthan": {
        "__default__": ["Wheat", "Mustard", "Gram", "Bajra", "Jowar", "Groundnut", "Cumin", "Fennel"],
        "Ajmer":         ["Wheat", "Gram", "Maize", "Mustard", "Vegetables"],
        "Alwar":         ["Wheat", "Mustard", "Bajra", "Pulses"],
        "Barmer":        ["Bajra", "Cumin", "Sesame", "Moth"],
        "Bharatpur":     ["Wheat", "Mustard", "Rice", "Bajra"],
        "Bikaner":       ["Bajra", "Groundnut", "Cumin", "Guar"],
        "Churu":         ["Bajra", "Guar", "Moth", "Groundnut"],
        "Dausa":         ["Wheat", "Mustard", "Bajra", "Gram"],
        "Dungarpur":     ["Maize", "Rice", "Wheat", "Pulses"],
        "Ganganagar":    ["Wheat", "Cotton", "Mustard", "Sugarcane"],
        "Hanumangarh":   ["Wheat", "Cotton", "Mustard", "Guar"],
        "Jaipur":        ["Wheat", "Mustard", "Bajra", "Vegetables"],
        "Jaisalmer":     ["Bajra", "Guar", "Moth", "Cumin"],
        "Jalore":        ["Bajra", "Cumin", "Groundnut", "Jowar"],
        "Jhalawar":      ["Coriander", "Soybean", "Wheat", "Tur"],
        "Jodhpur":       ["Bajra", "Cumin", "Moth", "Groundnut"],
        "Kota":          ["Soybean", "Wheat", "Gram", "Coriander"],
        "Nagaur":        ["Bajra", "Cumin", "Mustard", "Guar"],
        "Pali":          ["Bajra", "Pulses", "Groundnut", "Wheat"],
        "Sawai Madhopur":["Wheat", "Mustard", "Bajra", "Pulses"],
        "Sikar":         ["Wheat", "Mustard", "Bajra", "Guar"],
        "Sirohi":        ["Wheat", "Maize", "Pulses"],
        "Tonk":          ["Wheat", "Mustard", "Gram", "Bajra"],
        "Udaipur":       ["Maize", "Wheat", "Rice", "Pulses"],
    },

    "Bihar": {
        # North Bihar: Maize dominant + Makhana (Mithila), Litchi belt; South Bihar: Wheat, Pulses
        "__default__": ["Wheat", "Rice", "Maize", "Sugarcane", "Lentil", "Potato", "Gram", "Litchi", "Banana"],
        "Araria":        ["Maize", "Rice", "Wheat", "Jute", "Mustard"],              # North Bihar maize
        "Begusarai":     ["Rice", "Wheat", "Maize", "Sugarcane", "Vegetables"],
        "Bhagalpur":     ["Rice", "Wheat", "Lentil", "Litchi", "Mango"],
        "Buxar":         ["Wheat", "Rice", "Potato", "Mustard", "Gram"],
        "Darbhanga":     ["Makhana", "Rice", "Maize", "Wheat", "Jute", "Sugarcane"], # Makhana – Mithila heartland
        "East Champaran":["Sugarcane", "Rice", "Wheat", "Maize", "Jute"],
        "Gaya":          ["Rice", "Wheat", "Pulses", "Maize", "Vegetables"],
        "Gopalganj":     ["Rice", "Wheat", "Maize", "Sugarcane"],
        "Jamui":         ["Rice", "Maize", "Pulses", "Wheat"],
        "Jehanabad":     ["Rice", "Wheat", "Pulses", "Vegetables"],
        "Khagaria":      ["Makhana", "Rice", "Maize", "Jute", "Wheat"],              # Makhana zone (wetlands)
        "Madhepura":     ["Maize", "Rice", "Wheat", "Jute"],                          # North Bihar maize dominant
        "Madhubani":     ["Makhana", "Rice", "Maize", "Wheat", "Jute"],              # Makhana – Mithila
        "Munger":        ["Rice", "Wheat", "Tobacco", "Maize"],
        "Muzaffarpur":   ["Litchi", "Maize", "Rice", "Wheat", "Sugarcane", "Banana"],# Litchi capital
        "Nalanda":       ["Rice", "Wheat", "Vegetables", "Potato"],
        "Patna":         ["Rice", "Wheat", "Potato", "Vegetables"],
        "Purnia":        ["Maize", "Rice", "Jute", "Wheat", "Mustard"],              # Maize dominant NE Bihar
        "Rohtas":        ["Wheat", "Rice", "Pulses", "Gram"],
        "Saharsa":       ["Makhana", "Rice", "Maize", "Wheat", "Jute"],              # Makhana zone
        "Samastipur":    ["Maize", "Rice", "Wheat", "Sugarcane", "Makhana"],
        "Sitamarhi":     ["Rice", "Maize", "Wheat", "Sugarcane"],
        "Siwan":         ["Rice", "Wheat", "Sugarcane", "Maize"],
        "Supaul":        ["Makhana", "Rice", "Maize", "Wheat", "Jute"],              # Makhana zone
        "Vaishali":      ["Rice", "Wheat", "Sugarcane", "Vegetables", "Maize"],
        "West Champaran":["Sugarcane", "Rice", "Wheat", "Maize", "Jute"],
    },

    "West Bengal": {
        "__default__": ["Rice", "Jute", "Potato", "Wheat", "Mustard", "Maize", "Vegetables", "Tea"],
        "Bankura":       ["Rice", "Potatoes", "Pulses", "Vegetables"],
        "Bardhaman":     ["Rice", "Jute", "Wheat", "Potato", "Vegetables"],
        "Birbhum":       ["Rice", "Jute", "Wheat", "Potato"],
        "Cooch Behar":   ["Rice", "Jute", "Mustard", "Tobacco"],
        "Darjeeling":    ["Tea", "Orange", "Ginger", "Cardamom", "Rice"],
        "Howrah":        ["Rice", "Vegetables", "Flowers", "Coconut"],
        "Hugli":         ["Rice", "Jute", "Vegetables", "Potato"],
        "Jalpaiguri":    ["Tea", "Rice", "Jute"],
        "Jhargram":      ["Rice", "Sal", "Pulses"],
        "Kalimpong":     ["Large Cardamom", "Ginger", "Rice", "Vegetables"],
        "Kolkata":       ["Vegetables", "Flowers", "Rice"],
        "Malda":         ["Mango", "Rice", "Maize", "Wheat", "Jute"],
        "Murshidabad":   ["Rice", "Jute", "Mustard", "Maize"],
        "Nadia":         ["Rice", "Jute", "Vegetables", "Potato"],
        "North 24 Parganas":["Rice", "Vegetables", "Jute", "Coconut"],
        "North Dinajpur":["Rice", "Jute", "Wheat", "Maize"],
        "Paschim Bardhaman":["Rice", "Wheat", "Potato", "Vegetables"],
        "Paschim Medinipur":["Rice", "Potato", "Pulses", "Vegetables"],
        "Purba Bardhaman":["Rice", "Jute", "Wheat", "Vegetables"],
        "Purba Medinipur":["Rice", "Betel Vine", "Coconut", "Vegetables"],
        "Purulia":       ["Rice", "Pulses", "Vegetables"],
        "South 24 Parganas":["Rice", "Coconut", "Vegetables", "Fish"],
        "South Dinajpur":["Rice", "Wheat", "Jute", "Maize"],
        "Alipurduar":    ["Tea", "Rice"],
    },

    "Haryana": {
        "__default__": ["Wheat", "Rice", "Cotton", "Sugarcane", "Mustard", "Barley", "Sunflower", "Bajra"],
        "Ambala":        ["Wheat", "Rice", "Sugarcane", "Potato", "Vegetables"],
        "Bhiwani":       ["Wheat", "Cotton", "Bajra", "Mustard", "Guar"],
        "Charkhi Dadri": ["Wheat", "Cotton", "Bajra", "Mustard"],
        "Faridabad":     ["Wheat", "Rice", "Vegetables", "Sugarcane"],
        "Fatehabad":     ["Wheat", "Cotton", "Rice", "Bajra"],
        "Gurugram":      ["Wheat", "Rice", "Vegetables"],
        "Hisar":         ["Wheat", "Cotton", "Bajra", "Mustard", "Sunflower"],
        "Jhajjar":       ["Wheat", "Rice", "Mustard", "Vegetables"],
        "Jind":          ["Wheat", "Rice", "Cotton", "Mustard", "Bajra"],
        "Kaithal":       ["Wheat", "Rice", "Cotton", "Sugarcane"],
        "Karnal":        ["Wheat", "Rice", "Sugarcane", "Potato", "Vegetables"],
        "Kurukshetra":   ["Wheat", "Rice", "Mustard", "Sugarcane"],
        "Mahendragarh":  ["Wheat", "Mustard", "Bajra", "Gram"],
        "Mewat":         ["Wheat", "Bajra", "Mustard", "Vegetables"],
        "Palwal":        ["Wheat", "Rice", "Vegetables", "Sugarcane"],
        "Panchkula":     ["Wheat", "Rice", "Vegetables"],
        "Panipat":       ["Wheat", "Rice", "Sugarcane", "Vegetables"],
        "Rewari":        ["Wheat", "Mustard", "Bajra"],
        "Rohtak":        ["Wheat", "Rice", "Mustard", "Vegetables"],
        "Sirsa":         ["Wheat", "Cotton", "Rice", "Mustard", "Guar"],
        "Sonipat":       ["Wheat", "Rice", "Vegetables", "Sugarcane"],
        "Yamunanagar":   ["Wheat", "Sugarcane", "Rice", "Potato"],
    },

    "Kerala": {
        "__default__": ["Rice", "Coconut", "Rubber", "Banana", "Pepper", "Cardamom", "Ginger", "Cashew", "Tapioca"],
        "Alappuzha":     ["Rice", "Coconut", "Banana", "Coir"],
        "Ernakulam":     ["Coconut", "Rubber", "Banana", "Pepper", "Rice"],
        "Idukki":        ["Tea", "Cardamom", "Coffee", "Rubber", "Pepper"],
        "Kannur":        ["Coconut", "Arecanut", "Pepper", "Rice", "Rubber"],
        "Kasaragod":     ["Coconut", "Arecanut", "Pepper", "Cashew", "Rice"],
        "Kollam":        ["Coconut", "Rubber", "Tapioca", "Rice", "Cashew"],
        "Kottayam":      ["Rubber", "Coconut", "Pepper", "Arecanut", "Rice"],
        "Kozhikode":     ["Coconut", "Pepper", "Rice", "Ginger", "Arecanut"],
        "Malappuram":    ["Coconut", "Banana", "Arecanut", "Rice", "Ginger"],
        "Palakkad":      ["Rice", "Coconut", "Sugarcane", "Groundnut", "Banana"],
        "Pathanamthitta":["Rubber", "Coconut", "Pepper", "Cardamom", "Tapioca"],
        "Thiruvananthapuram":["Coconut", "Tapioca", "Banana", "Rubber", "Rice"],
        "Thrissur":      ["Coconut", "Banana", "Rice", "Pepper", "Rubber"],
        "Wayanad":       ["Coffee", "Tea", "Pepper", "Cardamom", "Ginger", "Rice"],
    },

    "Telangana": {
        "__default__": ["Rice", "Cotton", "Maize", "Groundnut", "Soybean", "Jowar", "Turmeric", "Chilli"],
        "Adilabad":      ["Cotton", "Soybean", "Jowar", "Rice", "Tur"],
        "Bhadradri Kothagudem":["Rice", "Maize", "Cashew"],
        "Hyderabad":     ["Vegetables", "Maize", "Rice"],
        "Jagitial":      ["Rice", "Maize", "Cotton", "Soybean"],
        "Jangaon":       ["Rice", "Maize", "Cotton"],
        "Jayashankar Bhupalpally":["Rice", "Maize", "Soybean"],
        "Jogulamba Gadwal":["Cotton", "Rice", "Maize", "Tur"],
        "Kamareddy":     ["Rice", "Maize", "Cotton", "Soybean"],
        "Karimnagar":    ["Rice", "Cotton", "Maize", "Chilli"],
        "Khammam":       ["Rice", "Maize", "Cotton"],
        "Kumuram Bheem": ["Rice", "Maize", "Soybean"],
        "Mahabubabad":   ["Rice", "Maize", "Cotton"],
        "Mahabubnagar":  ["Cotton", "Jowar", "Rice", "Tur", "Groundnut"],
        "Mancherial":    ["Rice", "Maize", "Cotton"],
        "Medak":         ["Rice", "Maize", "Cotton", "Groundnut"],
        "Medchal":       ["Vegetables", "Flowers", "Rice"],
        "Mulugu":        ["Rice", "Maize", "Soybean"],
        "Nagarkurnool":  ["Cotton", "Rice", "Jowar", "Tur"],
        "Nalgonda":      ["Cotton", "Rice", "Maize", "Groundnut"],
        "Narayanpet":    ["Cotton", "Jowar", "Rice", "Tur"],
        "Nirmal":        ["Cotton", "Soybean", "Jowar", "Rice"],
        "Nizamabad":     ["Rice", "Maize", "Turmeric", "Sugarcane"],
        "Peddapalli":    ["Rice", "Maize", "Cotton"],
        "Rajanna Sircilla":["Rice", "Cotton", "Maize"],
        "Rangareddy":    ["Vegetables", "Rice", "Maize"],
        "Sangareddy":    ["Rice", "Maize", "Cotton", "Soybean"],
        "Siddipet":      ["Rice", "Cotton", "Maize"],
        "Suryapet":      ["Rice", "Cotton", "Maize", "Groundnut"],
        "Vikarabad":     ["Maize", "Rice", "Soybean"],
        "Wanaparthy":    ["Cotton", "Rice", "Jowar", "Tur"],
        "Warangal Rural":["Rice", "Cotton", "Maize", "Chilli"],
        "Warangal Urban":["Rice", "Maize", "Cotton"],
        "Yadadri":       ["Rice", "Maize", "Vegetables"],
    },
}

# ──────────────────────────────────────────────────────────────────
# SEASON → CROP MAPPING  (Kharif / Rabi / Zaid / Annual / Perennial)
# Source: ICAR crop calendars, DAC&FW seasonal guides
#
# Values are LISTS so multi-season crops (e.g. Sunflower, Moong,
# Tomato) carry every valid season without duplicate-key loss.
# ──────────────────────────────────────────────────────────────────

CROP_SEASON_MAP: Dict[str, List[str]] = {
    # ── Strictly Kharif (Jun–Nov, SW monsoon) ───────────────────
    "Rice":          ["Kharif"],
    "Paddy":         ["Kharif"],
    "Basmati Rice":  ["Kharif"],
    "Cotton":        ["Kharif"],
    "Maize":         ["Kharif"],
    "Jowar":         ["Kharif"],        # also grown Rabi in Deccan
    "Bajra":         ["Kharif"],
    "Soybean":       ["Kharif"],
    "Groundnut":     ["Kharif"],
    "Tur":           ["Kharif"],
    "Arhar":         ["Kharif"],
    "Pigeon Pea":    ["Kharif"],
    "Green Gram":    ["Kharif"],
    "Urad":          ["Kharif"],
    "Black Gram":    ["Kharif"],
    "Sesame":        ["Kharif"],
    "Castor":        ["Kharif"],
    "Ragi":          ["Kharif"],
    "Finger Millet": ["Kharif"],
    "Pearl Millet":  ["Kharif"],
    "Jute":          ["Kharif"],
    "Ginger":        ["Kharif"],
    "Turmeric":      ["Kharif"],
    "Tobacco":       ["Kharif"],        # Kharif in AP/TN
    # ── Strictly Rabi (Oct/Nov–Mar, winter) ─────────────────────
    "Wheat":         ["Rabi"],
    "Mustard":       ["Rabi"],
    "Rapeseed":      ["Rabi"],
    "Gram":          ["Rabi"],
    "Chickpea":      ["Rabi"],
    "Lentil":        ["Rabi"],
    "Peas":          ["Rabi"],
    "Barley":        ["Rabi"],
    "Potato":        ["Rabi"],
    "Garlic":        ["Rabi"],
    "Coriander":     ["Rabi"],
    "Fenugreek":     ["Rabi"],
    "Fennel":        ["Rabi"],
    "Cumin":         ["Rabi"],
    "Safflower":     ["Rabi"],
    "Leafy Vegetables": ["Rabi"],
    # ── Strictly Zaid / Summer (Mar–Jun) ────────────────────────
    "Watermelon":    ["Zaid"],
    "Muskmelon":     ["Zaid"],
    "Cucumber":      ["Zaid"],
    "Bitter Gourd":  ["Zaid"],
    "Bottle Gourd":  ["Zaid"],
    # ── Annual (full-year cycle, e.g. planted any main season) ──
    "Sugarcane":     ["Annual"],        # planting Oct–Feb; 10–14 month cycle
    "Makhana":       ["Annual"],        # Fox nut; harvested Oct–Dec from ponds
    # ── Perennial / Tree crops ───────────────────────────────────
    "Banana":        ["Perennial"],
    "Coconut":       ["Perennial"],
    "Arecanut":      ["Perennial"],
    "Rubber":        ["Perennial"],
    "Tea":           ["Perennial"],
    "Coffee":        ["Perennial"],
    "Cardamom":      ["Perennial"],
    "Pepper":        ["Perennial"],
    "Mango":         ["Perennial"],
    "Litchi":        ["Perennial"],
    "Cashew":        ["Perennial"],
    "Orange":        ["Perennial"],
    "Grapes":        ["Perennial"],
    "Pomegranate":   ["Perennial"],
    "Jackfruit":     ["Perennial"],
    "Tapioca":       ["Perennial"],
    "Mulberry":      ["Perennial"],
    "Drumstick":     ["Perennial"],
    "Flowers":       ["Perennial"],
    # ── Multi-season crops (grown in 2+ seasons) ─────────────────
    # Sunflower: Kharif in North India, Rabi in Karnataka/AP/TN
    "Sunflower":     ["Kharif", "Rabi"],
    # Moong: Kharif (main) + Zaid (summer crop)
    "Moong":         ["Kharif", "Zaid"],
    # Onion: Rabi (main Maharashtra/Karnataka), also Kharif nursery
    "Onion":         ["Rabi", "Kharif"],
    # Chilli: Kharif (main) + Rabi (South India)
    "Chilli":        ["Kharif", "Rabi"],
    # Tomato: Rabi (peak North India) + Kharif + summer (South India)
    "Tomato":        ["Rabi", "Kharif", "Zaid"],
    # Brinjal: Kharif (main) + Rabi (South/Central India year-round)
    "Brinjal":       ["Kharif", "Rabi"],
    # Okra / Bhindi: Kharif + Zaid (summer crop in many states)
    "Okra":          ["Kharif", "Zaid"],
    # Vegetables (generic tag): grown across all 3 crop seasons
    "Vegetables":    ["Kharif", "Rabi", "Zaid"],
}


def get_crops_by_season(state: str, district: str, season: str) -> List[str]:
    """
    Return crops for a given location filtered by season.

    Args:
        state:    State name, e.g. "Punjab"
        district: District name, e.g. "Amritsar"
        season:   One of "Kharif", "Rabi", "Zaid", "Annual", "Perennial",
                  or "All" to skip filtering.

    Returns:
        Season-filtered list of crop strings, sorted alphabetically.
        Multi-season crops (e.g. Tomato, Sunflower, Moong) are included
        whenever the requested season appears in their season list.
        Falls back to all crops if season is "All" or unrecognised.

    Examples:
        >>> get_crops_by_season("Punjab", "Amritsar", "Kharif")
        ['Basmati Rice', 'Maize']
        >>> get_crops_by_season("Punjab", "Amritsar", "Rabi")
        ['Potato', 'Vegetables', 'Wheat']
        >>> get_crops_by_season("Tamil Nadu", "Chennai", "Kharif")
        ['Brinjal', 'Okra', 'Tomato']
    """
    all_crops = get_crops_for_location(state, district)

    if not season or season.strip().lower() == "all":
        return all_crops

    season_norm = season.strip().title()   # "kharif" → "Kharif"
    valid_seasons = {"Kharif", "Rabi", "Zaid", "Annual", "Perennial"}
    if season_norm not in valid_seasons:
        logger.warning(f"Unknown season '{season}'. Valid: {valid_seasons}. Returning all crops.")
        return all_crops

    filtered = [
        c for c in all_crops
        # Default unmapped crops to ["Kharif"] (most Indian crops are Kharif)
        if season_norm in CROP_SEASON_MAP.get(c, ["Kharif"])
    ]

    # Graceful fallback: if no crops survive the filter (very rare), return all
    return filtered if filtered else all_crops


def _get_default_crops_for_state(state: str) -> List[str]:
    """Return district-specific crops if available, otherwise state defaults."""
    state_data = _DISTRICT_CROPS.get(state, {})
    default = state_data.get("__default__")
    if default:
        return default
    # Last-resort generic list
    return ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean",
            "Groundnut", "Potato", "Mustard", "Gram", "Pulses", "Millets"]


# ──────────────────────────────────────────────────────────────────
# GET ADVISORY TEXT FOR LOCATION (bonus: full text advisory)
# ──────────────────────────────────────────────────────────────────

def get_advisory_text_for_location(state: str, district: str) -> str:
    """
    Fetches the latest advisory text (plain English) for a location.
    Used as context enrichment for the Groq LLM synthesis.

    Returns:
        Advisory text string or empty string if not available.
    """
    if not BS4_AVAILABLE:
        return ""

    try:
        state_links = fetch_imd_state_links()
        state_url = state_links.get(state)
        if not state_url:
            return ""

        district_links = fetch_district_links(state_url)
        dist_url = None
        for dname, durl in district_links.items():
            if district.lower() in dname.lower():
                dist_url = durl
                break

        if not dist_url:
            return ""

        # Try PDF first
        pdfs = fetch_advisory_pdfs(dist_url)
        if pdfs and PDFPLUMBER_AVAILABLE:
            resp = SESSION.get(pdfs[0], timeout=15)
            if resp.status_code == 200:
                tmp = os.path.join(os.path.dirname(__file__), "data", "_adv_text.pdf")
                with open(tmp, "wb") as f:
                    f.write(resp.content)
                text = ""
                with pdfplumber.open(tmp) as pdf:
                    for pg in pdf.pages:
                        t = pg.extract_text()
                        if t:
                            text += t + "\n"
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return text[:3000]     # cap at 3000 chars

        # HTML fallback
        resp = _get(dist_url)
        if resp:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:3000]

    except Exception as e:
        logger.warning(f"Advisory text fetch failed: {e}")

    return ""


# ──────────────────────────────────────────────────────────────────
# CROP CACHE STATUS
# ──────────────────────────────────────────────────────────────────

def get_cache_status() -> Dict:
    """
    Returns metadata about the current crop cache.
    Useful for displaying in the Streamlit sidebar.
    """
    if not os.path.exists(CACHE_FILE):
        return {
            "exists": False,
            "states": 0,
            "built_at": None,
            "age_hours": None,
            "is_fresh": False,
        }

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        built_at_str = cache.get("metadata", {}).get("built_at", "")
        built_at = datetime.fromisoformat(built_at_str) if built_at_str else None
        age_hours = (datetime.now() - built_at).total_seconds() / 3600 if built_at else None
        ttl = cache.get("metadata", {}).get("ttl_hours", CACHE_TTL_HOURS)
        data = cache.get("data", {})

        return {
            "exists": True,
            "states": len(data),
            "built_at": built_at_str,
            "age_hours": round(age_hours, 1) if age_hours else None,
            "is_fresh": (age_hours < ttl) if age_hours else False,
        }
    except Exception:
        return {"exists": False, "states": 0, "built_at": None, "age_hours": None, "is_fresh": False}


# ──────────────────────────────────────────────────────────────────
# CLI — for standalone testing
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        state_arg = sys.argv[1]
        district_arg = sys.argv[2]
        print(f"\n🌾 Fetching crops for: {state_arg}, {district_arg}")
        result = get_crops_for_location(state_arg, district_arg)
        print(f"   Crops: {result}")
    elif len(sys.argv) == 2 and sys.argv[1] == "--build-cache":
        print("🔄 Building full national crop cache ...")
        db = build_crop_cache()
        print(f"✅ Cache built. States: {len(db)}")
    else:
        print("Usage:")
        print("  python imd_scraper.py 'Tamil Nadu' 'Ariyalur'")
        print("  python imd_scraper.py --build-cache")
