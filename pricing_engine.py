"""
pricing_engine.py
------------------
Authoritative, server-side subscription pricing for Eduket OS.

    base_zar     = BASE_TIER_PRICES_ZAR[tier][cycle] * INSTITUTION_MULTIPLIERS[institution_type]
    charge_amount = base_zar * fx_rate(registrant_currency) * currency_multiplier(registrant_currency)
    charge_currency = registrant_currency

`currency_multiplier` is 3 if the registrant's local currency is "stronger"
than ZAR (one unit of that currency is worth more than one ZAR), otherwise 1.
Since PayFast multi-currency settlement is active on this account, that
final amount is charged directly in `charge_currency` - there's no more
"ZAR-only with a local estimate" step. `amount_zar_equivalent` is still
returned alongside it purely for your own ZAR-denominated revenue
reporting; it is NOT what gets sent to PayFast.

THIS FILE IS THE SOURCE OF TRUTH FOR PRICE. The frontend should never
compute a chargeable amount itself - it calls /api/billing/quote (read-only
preview) or /api/billing/initiate (creates the authoritative pending
transaction right before redirecting to PayFast) and uses whatever this
module returns. See billing_routes.py.

KEY ASSUMPTIONS / OPEN ITEMS - please confirm or tell me to change these:

1. PayFast's exact parameter for specifying a non-ZAR settlement currency
   per transaction. I don't have confident knowledge of PayFast's specific
   field name for this (it might be a `currency` form field, a separate
   merchant ID per currency, or an account-level setting rather than a
   per-transaction parameter) - I've used `PAYFAST_CURRENCY_FIELD` as a
   single, clearly-isolated guess in billing_routes.py. Check your PayFast
   multi-currency integration docs/dashboard and correct that one spot if
   I guessed wrong - getting this wrong could mean a payment silently
   settles in the wrong currency or fails outright.

2. Institution levels compound: secondary = 1.5x primary, tertiary
   (university/college) = 1.5x secondary = 2.25x primary. If you actually
   meant a flat +50% of the primary price at every level instead
   (tertiary = 2.0x primary), change TERTIARY below to 2.0.

3. "Stronger currency" is judged purely by FX unit-value vs ZAR, exactly as
   you described it. One side effect: a few large, wealthy economies whose
   currency has a low face value per unit (Japan/JPY, South Korea/KRW,
   Vietnam/VND, Indonesia/IDR) will land in the "normal rate" bucket
   despite being rich markets. If you'd rather classify by economic
   strength (e.g. a maintained list of high-income countries) instead of
   raw FX unit value, say so and I'll swap out
   `is_currency_stronger_than_zar`'s implementation - everything else stays
   the same.

4. `country_raw` can be a full country name ("South Africa"), a common
   alias ("USA", "South Korea", "Ivory Coast"), or an ISO alpha-2/alpha-3
   code. `resolve_country_to_alpha2`/`resolve_country_to_currency` handle
   all three. Confirmed against your actual schools schema: `countryCode`
   (ISO alpha-2) is read ahead of `country` (full name) in
   billing_routes.py, and `teachingPhases` (an array) is resolved directly
   by `resolve_institution_type`, billing at the highest phase present if a
   school spans more than one.

5. Diamond's limits (students/teachers/exams) are set to double Platinum's,
   matching the price-doubling instruction - you only specified the price,
   so I extended the same multiplier to the limits for consistency. Easy to
   change, they're plain numbers in BASE_TIER_PRICES_ZAR / TIER_EXAM_LIMITS
   and TIERS in tierConfig.js.

6. DOLLARIZED_ECONOMY_OVERRIDES exists (Zimbabwe, Ecuador, El Salvador, and
   a few small Pacific/Caribbean territories all resolve to USD without
   being wealthy markets) but is currently DISABLED
   (ENABLE_DOLLARIZED_ECONOMY_OVERRIDE = False) - confirmed: USD-currency
   regions bill as a strong currency with no exceptions, Zimbabwe included.
   Flip the toggle back to True if that ever changes.
"""

import logging
import time
from datetime import datetime, timezone

import requests

try:
    import pycountry
except ImportError:  # pragma: no cover
    pycountry = None

try:
    from firebase_admin import firestore
except ImportError:  # pragma: no cover - keeps this importable without firebase_admin installed
    firestore = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Base tier pricing (ZAR). Keep this identical to TIERS in
#    frontend/src/utils/tierConfig.js - if a price changes there, change it
#    here too.
# ---------------------------------------------------------------------------
BASE_TIER_PRICES_ZAR = {
    "free": {"monthly": 0, "annual": 0},
    "silver": {"monthly": 799, "annual": 7990},
    "gold": {"monthly": 1399, "annual": 13990},
    "platinum": {"monthly": 2999, "annual": 29990},
    "diamond": {"monthly": 5998, "annual": 59980},  # = 2x platinum, as requested
}

# ---------------------------------------------------------------------------
# 2. Institution-level multipliers
# ---------------------------------------------------------------------------
PRIMARY, SECONDARY, TERTIARY = "primary", "secondary", "tertiary"

INSTITUTION_MULTIPLIERS = {
    PRIMARY: 1.0,
    SECONDARY: 1.5,   # +50% of primary
    TERTIARY: 2.25,   # +50% of secondary (university / college) - see assumption #2 above
}


def normalize_institution_type(raw):
    """Map whatever string your school doc stores to one of the 3 buckets above.
    Defaults to PRIMARY (the cheapest bucket) when unknown, rather than guessing high."""
    if not raw:
        return PRIMARY
    value = str(raw).strip().lower()
    if any(k in value for k in ("university", "college", "tertiary", "higher")):
        return TERTIARY
    if any(k in value for k in ("secondary", "high school", "high")):
        return SECONDARY
    return PRIMARY


def get_institution_multiplier(institution_type):
    return INSTITUTION_MULTIPLIERS.get(normalize_institution_type(institution_type), 1.0)


INSTITUTION_RANK = {PRIMARY: 0, SECONDARY: 1, TERTIARY: 2}


def resolve_institution_type(raw):
    """Accepts either a single string OR a list (your schools store
    `teachingPhases` as an array, e.g. ["Primary", "Secondary"]) and
    returns the single highest-billing bucket present. A school spanning
    Primary and Secondary is billed at the Secondary rate - the more
    expensive level it actually offers, not the cheapest."""
    if raw is None:
        return PRIMARY
    if isinstance(raw, (list, tuple, set)):
        if not raw:
            return PRIMARY
        candidates = [normalize_institution_type(item) for item in raw]
        return max(candidates, key=lambda c: INSTITUTION_RANK[c])
    return normalize_institution_type(raw)


# ---------------------------------------------------------------------------
# 3. Country -> currency resolution
#    ALPHA2_TO_CURRENCY was generated from babel's CLDR territory-currency
#    data (the same data Unicode/ICU uses) - it is not hand-typed, so treat
#    it as reliable. COUNTRY_NAME_ALIASES patches the handful of common
#    names that pycountry's own lookup/fuzzy-search can't resolve
#    (verified empirically: "Ivory Coast", "Swaziland", "Burma", DRC
#    variants, "Macau").
# ---------------------------------------------------------------------------
ALPHA2_TO_CURRENCY = {
    "AD": "EUR", "AE": "AED", "AF": "AFN", "AG": "XCD", "AI": "XCD", "AL": "ALL",
    "AM": "AMD", "AO": "AOA", "AR": "ARS", "AS": "USD", "AT": "EUR", "AU": "AUD",
    "AW": "AWG", "AX": "EUR", "AZ": "AZN", "BA": "BAM", "BB": "BBD", "BD": "BDT",
    "BE": "EUR", "BF": "XOF", "BG": "BGN", "BH": "BHD", "BI": "BIF", "BJ": "XOF",
    "BL": "EUR", "BM": "BMD", "BN": "BND", "BO": "BOB", "BQ": "USD", "BR": "BRL",
    "BS": "BSD", "BT": "INR", "BV": "NOK", "BW": "BWP", "BY": "BYN", "BZ": "BZD",
    "CA": "CAD", "CC": "AUD", "CD": "CDF", "CF": "XAF", "CG": "XAF", "CH": "CHF",
    "CI": "XOF", "CK": "NZD", "CL": "CLP", "CM": "XAF", "CN": "CNY", "CO": "COP",
    "CR": "CRC", "CU": "CUP", "CV": "CVE", "CW": "XCG", "CX": "AUD", "CY": "EUR",
    "CZ": "CZK", "DE": "EUR", "DJ": "DJF", "DK": "DKK", "DM": "XCD", "DO": "DOP",
    "DZ": "DZD", "EC": "USD", "EE": "EUR", "EG": "EGP", "EH": "MAD", "ER": "ERN",
    "ES": "EUR", "ET": "ETB", "FI": "EUR", "FJ": "FJD", "FK": "FKP", "FM": "USD",
    "FO": "DKK", "FR": "EUR", "GA": "XAF", "GB": "GBP", "GD": "XCD", "GE": "GEL",
    "GF": "EUR", "GG": "GBP", "GH": "GHS", "GI": "GIP", "GL": "DKK", "GM": "GMD",
    "GN": "GNF", "GP": "EUR", "GQ": "XAF", "GR": "EUR", "GS": "GBP", "GT": "GTQ",
    "GU": "USD", "GW": "XOF", "GY": "GYD", "HK": "HKD", "HM": "AUD", "HN": "HNL",
    "HR": "EUR", "HT": "HTG", "HU": "HUF", "ID": "IDR", "IE": "EUR", "IL": "ILS",
    "IM": "GBP", "IN": "INR", "IO": "USD", "IQ": "IQD", "IR": "IRR", "IS": "ISK",
    "IT": "EUR", "JE": "GBP", "JM": "JMD", "JO": "JOD", "JP": "JPY", "KE": "KES",
    "KG": "KGS", "KH": "KHR", "KI": "AUD", "KM": "KMF", "KN": "XCD", "KP": "KPW",
    "KR": "KRW", "KW": "KWD", "KY": "KYD", "KZ": "KZT", "LA": "LAK", "LB": "LBP",
    "LC": "XCD", "LI": "CHF", "LK": "LKR", "LR": "LRD", "LS": "ZAR", "LT": "EUR",
    "LU": "EUR", "LV": "EUR", "LY": "LYD", "MA": "MAD", "MC": "EUR", "MD": "MDL",
    "ME": "EUR", "MF": "EUR", "MG": "MGA", "MH": "USD", "MK": "MKD", "ML": "XOF",
    "MM": "MMK", "MN": "MNT", "MO": "MOP", "MP": "USD", "MQ": "EUR", "MR": "MRU",
    "MS": "XCD", "MT": "EUR", "MU": "MUR", "MV": "MVR", "MW": "MWK", "MX": "MXN",
    "MY": "MYR", "MZ": "MZN", "NA": "ZAR", "NC": "XPF", "NE": "XOF", "NF": "AUD",
    "NG": "NGN", "NI": "NIO", "NL": "EUR", "NO": "NOK", "NP": "NPR", "NR": "AUD",
    "NU": "NZD", "NZ": "NZD", "OM": "OMR", "PA": "PAB", "PE": "PEN", "PF": "XPF",
    "PG": "PGK", "PH": "PHP", "PK": "PKR", "PL": "PLN", "PM": "EUR", "PN": "NZD",
    "PR": "USD", "PS": "ILS", "PT": "EUR", "PW": "USD", "PY": "PYG", "QA": "QAR",
    "RE": "EUR", "RO": "RON", "RS": "RSD", "RU": "RUB", "RW": "RWF", "SA": "SAR",
    "SB": "SBD", "SC": "SCR", "SD": "SDG", "SE": "SEK", "SG": "SGD", "SH": "SHP",
    "SI": "EUR", "SJ": "NOK", "SK": "EUR", "SL": "SLE", "SM": "EUR", "SN": "XOF",
    "SO": "SOS", "SR": "SRD", "SS": "SSP", "ST": "STN", "SV": "USD", "SX": "XCG",
    "SY": "SYP", "SZ": "SZL", "TC": "USD", "TD": "XAF", "TF": "EUR", "TG": "XOF",
    "TH": "THB", "TJ": "TJS", "TK": "NZD", "TL": "USD", "TM": "TMT", "TN": "TND",
    "TO": "TOP", "TR": "TRY", "TT": "TTD", "TV": "AUD", "TW": "TWD", "TZ": "TZS",
    "UA": "UAH", "UG": "UGX", "UM": "USD", "US": "USD", "UY": "UYU", "UZ": "UZS",
    "VA": "EUR", "VC": "XCD", "VE": "VES", "VG": "USD", "VI": "USD", "VN": "VND",
    "VU": "VUV", "WF": "XPF", "WS": "WST", "YE": "YER", "YT": "EUR", "ZA": "ZAR",
    "ZM": "ZMW", "ZW": "USD",
}

# Names/aliases that pycountry's lookup() and search_fuzzy() do NOT resolve
# (confirmed by testing), keyed lowercase.
COUNTRY_NAME_ALIASES = {
    "ivory coast": "CI",
    "swaziland": "SZ",
    "burma": "MM",
    "democratic republic of the congo": "CD",
    "dr congo": "CD",
    "congo-kinshasa": "CD",
    "republic of congo": "CG",
    "congo-brazzaville": "CG",
    "macau": "MO",
    "cape verde": "CV",
}

DEFAULT_CURRENCY = "ZAR"

# Countries where the official/CLDR currency is a "strong" foreign currency
# NOT because the local economy is wealthy, but because the country's own
# currency partially or fully collapsed and a foreign currency (almost
# always USD) is used instead - Zimbabwe being the clearest example.
#
# DISABLED per explicit instruction: USD-currency regions bill as a strong
# currency with no exceptions, Zimbabwe included. The list and toggle are
# left in place rather than deleted in case that decision ever gets
# revisited - flip ENABLE_DOLLARIZED_ECONOMY_OVERRIDE back to True to
# reinstate it, no other code changes needed.
ENABLE_DOLLARIZED_ECONOMY_OVERRIDE = False
DOLLARIZED_ECONOMY_OVERRIDES = {
    "ZW",  # Zimbabwe - multi-currency regime after the Zimbabwean dollar collapsed
    "EC",  # Ecuador
    "SV",  # El Salvador
    "TL",  # Timor-Leste
    "FM", "MH", "PW",  # Micronesia, Marshall Islands, Palau - USD by treaty, not market wealth
    "BQ", "TC",  # Caribbean territories settling in USD
}
# Note: Zimbabwe introduced a new currency (ZiG) in 2024. babel's data
# resolving ZW to USD may reflect real on-the-ground usage (USD has stayed
# widely used/preferred) or may simply lag a very recent change - worth
# confirming which currency you actually intend to bill Zimbabwean schools
# in, independent of the (now disabled) override above.


def resolve_country_to_alpha2(country_raw):
    """Best-effort resolution of a country string (name, alias, or ISO
    alpha-2/alpha-3 code) to an ISO 3166-1 alpha-2 code. Returns None if it
    can't be resolved."""
    if not country_raw:
        return None

    value = str(country_raw).strip()
    if not value:
        return None

    # Already an alpha-2 code?
    if len(value) == 2 and value.isalpha():
        code = value.upper()
        return code if code in ALPHA2_TO_CURRENCY else None

    lowered = value.lower()
    if lowered in COUNTRY_NAME_ALIASES:
        return COUNTRY_NAME_ALIASES[lowered]

    if pycountry is None:
        logger.warning("pycountry not installed - cannot resolve country name %r", country_raw)
        return None

    try:
        return pycountry.countries.lookup(value).alpha_2
    except LookupError:
        pass

    try:
        results = pycountry.countries.search_fuzzy(value)
        if results:
            return results[0].alpha_2
    except LookupError:
        pass

    logger.warning("Could not resolve country %r to an ISO code - defaulting to ZAR", country_raw)
    return None


def resolve_country_to_currency(country_raw):
    """Best-effort resolution of a country string to an ISO 4217 currency
    code. Returns None if it can't be resolved - callers should treat None
    as "default to ZAR / normal rate", not crash."""
    alpha2 = resolve_country_to_alpha2(country_raw)
    if alpha2 is None:
        return None
    return ALPHA2_TO_CURRENCY.get(alpha2)


# ---------------------------------------------------------------------------
# 4. Exchange rates: live fetch -> Firestore cache -> static fallback
#
#    Convention used throughout: rate = how many units of `currency` equal
#    1 ZAR. Example: USD rate ~0.054 (1 ZAR ~ 0.054 USD), so rate < 1 means
#    the foreign currency is "stronger" (each of its units is worth more
#    than 1 ZAR).
# ---------------------------------------------------------------------------
EXCHANGE_RATE_API_URL = "https://open.er-api.com/v6/latest/ZAR"
RATE_CACHE_COLLECTION = "system_config"
RATE_CACHE_DOC_ID = "exchangeRates"
RATE_CACHE_TTL_HOURS = 24
HTTP_TIMEOUT_SECONDS = 6

# Approximate fallback rates (units of currency per 1 ZAR). Used ONLY if both
# the live API and the Firestore cache are unavailable. These are NOT pulled
# live - review/update periodically. (Rough mid-2025 figures.)
FALLBACK_RATES_PER_ZAR = {
    "ZAR": 1.0,
    "USD": 0.054, "GBP": 0.043, "EUR": 0.050, "AUD": 0.083, "CAD": 0.074,
    "CHF": 0.048, "NZD": 0.090, "SGD": 0.073, "AED": 0.198, "HKD": 0.420,
    "JPY": 8.40, "KRW": 73.0, "CNY": 0.39, "INR": 4.55, "BRL": 0.31,
    "NGN": 84.0, "KES": 7.00, "EGP": 2.65, "GHS": 0.80,
    "NAD": 1.0, "LSL": 1.0, "SZL": 1.0, "BWP": 0.74,
}

_rate_cache = {"rates": None, "fetched_at": 0.0, "source": None}

# ISO 4217 minor units. Most currencies use 2 decimal places; a handful use
# 0 or 3. Used to round/format the actual charge amount correctly - this
# matters now that charges happen in the real currency, not just ZAR.
CURRENCY_ZERO_DECIMAL = {
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW",
    "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF",
}
CURRENCY_THREE_DECIMAL = {"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"}


def get_currency_decimals(currency_code):
    code = (currency_code or "").upper()
    if code in CURRENCY_ZERO_DECIMAL:
        return 0
    if code in CURRENCY_THREE_DECIMAL:
        return 3
    return 2


def round_for_currency(amount, currency_code):
    return round(amount, get_currency_decimals(currency_code))


def _fetch_live_rates():
    response = requests.get(EXCHANGE_RATE_API_URL, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    rates = payload.get("rates")
    if not rates or "USD" not in rates:
        raise ValueError("Unexpected exchange rate API response shape")
    return rates


def _read_firestore_cache():
    if firestore is None:
        return None
    try:
        db = firestore.client()
        snap = db.collection(RATE_CACHE_COLLECTION).document(RATE_CACHE_DOC_ID).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        rates = data.get("rates")
        fetched_at = data.get("fetchedAt")
        if not rates or fetched_at is None:
            return None
        age_hours = (time.time() - fetched_at) / 3600
        if age_hours > RATE_CACHE_TTL_HOURS:
            return None
        return rates
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read exchange rate cache from Firestore: %s", exc)
        return None


def _write_firestore_cache(rates):
    if firestore is None:
        return
    try:
        db = firestore.client()
        db.collection(RATE_CACHE_COLLECTION).document(RATE_CACHE_DOC_ID).set({
            "rates": rates,
            "fetchedAt": time.time(),
            "updatedAtIso": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write exchange rate cache to Firestore: %s", exc)


def _get_rates_with_source():
    """Returns (rates_dict, source) where source is 'memory' | 'firestore' | 'live' | 'fallback'."""
    now = time.time()
    if _rate_cache["rates"] and (now - _rate_cache["fetched_at"]) / 3600 < RATE_CACHE_TTL_HOURS:
        return _rate_cache["rates"], "memory"

    cached = _read_firestore_cache()
    if cached:
        _rate_cache.update(rates=cached, fetched_at=now, source="firestore")
        return cached, "firestore"

    try:
        live = _fetch_live_rates()
        _rate_cache.update(rates=live, fetched_at=now, source="live")
        _write_firestore_cache(live)
        return live, "live"
    except Exception as exc:  # noqa: BLE001
        logger.error("Exchange rate fetch failed, using static fallback table: %s", exc)
        return FALLBACK_RATES_PER_ZAR, "fallback"


def get_exchange_rates():
    rates, _source = _get_rates_with_source()
    return rates


def is_currency_stronger_than_zar(currency_code, rates=None):
    """True if 1 unit of currency_code is worth MORE than 1 ZAR."""
    if not currency_code or currency_code.upper() == "ZAR":
        return False
    rates = rates if rates is not None else get_exchange_rates()
    rate = rates.get(currency_code.upper())
    if rate is None:
        # Unknown currency - don't guess, default to normal (non-multiplied) pricing.
        return False
    return rate < 1


def get_currency_multiplier(currency_code, rates=None):
    return 3 if is_currency_stronger_than_zar(currency_code, rates=rates) else 1


# ---------------------------------------------------------------------------
# 5. Putting it together
# ---------------------------------------------------------------------------
def compute_subscription_price(tier_id, billing_cycle, institution_type, country_raw):
    """Returns the authoritative price breakdown for a subscription.
    `charge_amount` / `charge_currency` are what should actually be sent to
    PayFast. `amount_zar_equivalent` is informational only (ZAR-denominated
    revenue reporting) - it is NOT charged. `institution_type` may be a
    single string or a list (e.g. a teachingPhases array)."""
    cycle = "annual" if str(billing_cycle).lower().startswith("ann") else "monthly"
    tier_prices = BASE_TIER_PRICES_ZAR.get(tier_id, BASE_TIER_PRICES_ZAR["free"])
    base_amount = tier_prices[cycle]

    institution_normalized = resolve_institution_type(institution_type)
    institution_multiplier = INSTITUTION_MULTIPLIERS[institution_normalized]
    base_amount_zar = base_amount * institution_multiplier

    alpha2 = resolve_country_to_alpha2(country_raw)
    currency_code = (ALPHA2_TO_CURRENCY.get(alpha2) if alpha2 else None) or DEFAULT_CURRENCY
    rates, rate_source = _get_rates_with_source()

    if ENABLE_DOLLARIZED_ECONOMY_OVERRIDE and alpha2 in DOLLARIZED_ECONOMY_OVERRIDES:
        currency_multiplier = 1
    else:
        currency_multiplier = get_currency_multiplier(currency_code, rates=rates)

    fx_rate = rates.get(currency_code)
    if fx_rate is None:
        # Couldn't price in their currency at all (unknown/unsupported code) -
        # fall back to ZAR rather than charging an unconverted number in the
        # wrong currency.
        currency_code = DEFAULT_CURRENCY
        fx_rate = 1.0

    charge_amount = round_for_currency(base_amount_zar * fx_rate * currency_multiplier, currency_code)
    amount_zar_equivalent = round(base_amount_zar * currency_multiplier, 2)

    return {
        "tier_id": tier_id,
        "billing_cycle": cycle,
        "base_amount_zar": base_amount_zar,
        "institution_type": institution_normalized,
        "institution_multiplier": institution_multiplier,
        "currency_multiplier": currency_multiplier,
        "exchange_rate_source": rate_source,
        "charge_currency": currency_code,        # <-- pass to PayFast
        "charge_amount": charge_amount,           # <-- pass to PayFast
        "amount_zar_equivalent": amount_zar_equivalent,  # reporting only, not charged
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scenarios = [
        ("gold", "monthly", "primary", "South Africa"),
        ("gold", "monthly", "secondary", "South Africa"),
        ("gold", "monthly", "tertiary", "South Africa"),
        ("gold", "monthly", "primary", "United States"),
        ("gold", "monthly", "tertiary", "United States"),
        ("gold", "monthly", "primary", "United Kingdom"),
        ("gold", "monthly", "primary", "Nigeria"),
        ("gold", "monthly", "primary", "South Korea"),
        ("gold", "monthly", "primary", "Ivory Coast"),
        ("silver", "annual", "secondary", "Kenya"),
        ("platinum", "monthly", "tertiary", "USA"),
        ("diamond", "monthly", "primary", "South Africa"),
        ("diamond", "monthly", "tertiary", "United States"),
        ("diamond", "annual", "primary", "Japan"),
        ("gold", "monthly", ["Secondary"], "ZW"),               # Prince Edward High, real data
        ("gold", "monthly", ["Primary", "Secondary"], "Kenya"),  # multi-phase school -> bills as Secondary
    ]
    for tier, cycle, inst, country in scenarios:
        result = compute_subscription_price(tier, cycle, inst, country)
        print(
            f"{tier:9s} {cycle:8s} {str(inst):28s} {country:16s} -> "
            f"{result['charge_currency']} {result['charge_amount']:>12,.2f}  "
            f"(inst={result['institution_type']} x{result['institution_multiplier']}, currency x{result['currency_multiplier']}, "
            f"ZAR-equiv R{result['amount_zar_equivalent']:,.2f}, rates={result['exchange_rate_source']})"
        )