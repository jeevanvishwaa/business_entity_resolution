"""
High-performance text normalization module for business names, addresses, and country labels.
Phase 6 Promoted Production Configuration:
- Multilingual character normalization and NFKD unicode stripping
- Legal suffix removal and core business tokenization
- Country-Conditioned Address Normalization (P5-02 Promoted):
  * US: TN -> Tennessee, NY -> New York, TX -> Texas, etc.
  * India: TN -> Tamil Nadu, DL -> Delhi, KA -> Karnataka, etc.
  * France: bd/bvd -> boulevard, av -> avenue, r -> rue, rte -> route, etc.
"""

import re
import unicodedata

# Common legal suffixes across US, India, France, and Global entities
LEGAL_SUFFIXES = {
    'corporation', 'corp', 'incorporated', 'inc', 'limited', 'ltd', 'pvt', 'private',
    'llc', 'llp', 'pllc', 'company', 'co', 'gmbh', 'sa', 'sarl', 'sas', 'sasu', 'snc',
    'eurl', 'sci', 'enterprises', 'enterprise', 'services', 'service', 'solutions',
    'solution', 'holdings', 'holding', 'group', 'industries', 'industry'
}

COMMON_ADDRESS_ABBREVIATIONS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane', 'ct': 'court',
    'pl': 'place', 'sq': 'square', 'ste': 'suite', 'apt': 'apartment',
    'fl': 'floor', 'pkwy': 'parkway', 'hwy': 'highway', 'str': 'street',
    'bldg': 'building'
}

US_STATE_ABBREVIATIONS = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland', 'ma': 'massachusetts',
    'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi', 'mo': 'missouri', 'mt': 'montana',
    'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico',
    'ny': 'new york', 'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah', 'vt': 'vermont',
    'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming'
}

INDIA_STATE_ABBREVIATIONS = {
    'tn': 'tamil nadu', 'up': 'uttar pradesh', 'mp': 'madhya pradesh', 'ap': 'andhra pradesh',
    'mh': 'maharashtra', 'ka': 'karnataka', 'dl': 'delhi', 'kl': 'kerala', 'wb': 'west bengal',
    'ts': 'telangana', 'gj': 'gujarat', 'rj': 'rajasthan', 'pb': 'punjab', 'hr': 'haryana',
    'opp': 'opposite', 'nr': 'near'
}

FRANCE_ABBREVIATIONS = {
    'bd': 'boulevard', 'bvd': 'boulevard', 'av': 'avenue', 'r': 'rue', 'rte': 'route',
    'imp': 'impasse', 'all': 'allee', 'pl': 'place', 'fg': 'faubourg'
}

# Regex pre-compilations for speed
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_NON_ALPHANUM_MULTI = re.compile(r'[^\w\s]', re.UNICODE)
RE_WHITESPACE = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_POSTAL = re.compile(r'\b\d{5,6}\b')

def normalize_text_basic(text: str) -> str:
    """Basic unicode normalization, lowercase, and whitespace stripping."""
    if not text or not isinstance(text, str) or text.lower() in ('nan', 'none', 'null', 'na'):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.replace('&', ' and ')
    text = text.replace('/', ' ')
    text = text.replace('-', ' ')
    text = text.replace('_', ' ')
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.lower()
    return RE_WHITESPACE.sub(' ', text).strip()

def normalize_country(country: str) -> str:
    """Standardize country label to canonical token."""
    if not country or not isinstance(country, str):
        return ""
    c = country.strip().upper()
    if 'US' in c or 'UNITED STATES' in c or 'USA' in c:
        return 'US'
    if 'IN' in c or 'INDIA' in c:
        return 'INDIA'
    if 'FR' in c or 'FRANCE' in c:
        return 'FRANCE'
    return c

def normalize_business_name(name: str) -> dict:
    """Returns clean name, compact name, distinctive tokens, and core tokens."""
    raw_clean = normalize_text_basic(name)
    if not raw_clean:
        return {
            'raw_clean': '',
            'compact': '',
            'tokens': [],
            'core_tokens': [],
            'prefix_key': '',
            'distinctive_tokens': []
        }

    clean_no_punct = RE_NON_ALPHANUM_MULTI.sub('', raw_clean)
    tokens = clean_no_punct.split()
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES and t not in ('and', 'the', 'of', 'in', 'at')]
    if not core_tokens:
        core_tokens = tokens

    compact = ''.join(tokens)
    prefix_key = ' '.join(core_tokens[:2]) if len(core_tokens) >= 2 else (core_tokens[0] if core_tokens else '')
    distinctive_tokens = [t for t in core_tokens if len(t) >= 4 and not t.isdigit()]

    return {
        'raw_clean': raw_clean,
        'compact': compact,
        'tokens': tokens,
        'core_tokens': core_tokens,
        'prefix_key': prefix_key,
        'distinctive_tokens': distinctive_tokens
    }

def country_aware_normalize_address(address: str, country: str = "") -> dict:
    """Country-conditioned address normalization (Phase 6 Promoted)."""
    raw_clean = normalize_text_basic(address)
    if not raw_clean:
        return {
            'raw_clean': '',
            'compact': '',
            'numbers': [],
            'postal_code': '',
            'street_number': '',
            'tokens': []
        }

    c_norm = normalize_country(country) if country else ""
    tokens = raw_clean.split()

    expanded_tokens = []
    for t in tokens:
        if t in COMMON_ADDRESS_ABBREVIATIONS:
            expanded_tokens.append(COMMON_ADDRESS_ABBREVIATIONS[t])
        elif c_norm == 'US' and t in US_STATE_ABBREVIATIONS:
            expanded_tokens.append(US_STATE_ABBREVIATIONS[t])
        elif c_norm == 'INDIA' and t in INDIA_STATE_ABBREVIATIONS:
            expanded_tokens.append(INDIA_STATE_ABBREVIATIONS[t])
        elif c_norm == 'FRANCE' and t in FRANCE_ABBREVIATIONS:
            expanded_tokens.append(FRANCE_ABBREVIATIONS[t])
        else:
            expanded_tokens.append(t)

    clean_addr = ' '.join(expanded_tokens)
    numbers = RE_NUMBERS.findall(clean_addr)
    postal_matches = RE_POSTAL.findall(clean_addr)
    postal_code = postal_matches[0] if postal_matches else ''
    street_number = numbers[0] if numbers else ''

    clean_tokens = RE_NON_ALPHANUM_MULTI.sub('', clean_addr).split()
    compact = ''.join(clean_tokens)

    return {
        'raw_clean': clean_addr,
        'compact': compact,
        'numbers': numbers,
        'postal_code': postal_code,
        'street_number': street_number,
        'tokens': clean_tokens
    }

def normalize_address(address: str, country: str = "") -> dict:
    """Primary address normalization function delegating to country-aware normalization."""
    return country_aware_normalize_address(address, country=country)
