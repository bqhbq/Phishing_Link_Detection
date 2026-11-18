"""Chuẩn hóa URL và trích xuất đặc trưng (kế thừa notebook 06, đóng gói thành module).

Nguyên tắc:
  * MỘT hàm `build_feature_frame` dùng chung cho TRAIN và INFERENCE → không còn train/serve skew.
  * Registry đặc trưng tường minh (`FEATURE_GROUPS`) thay cho auto-discovery `module.__dict__` của code cũ.
  * Chỉ phân tích chuỗi (offline): không gọi mạng, tldextract dùng Public Suffix List đóng gói sẵn.
"""
from __future__ import annotations

import importlib
import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# --------------------------------------------------------------------------------------
# 1. Chuẩn hóa URL (canonicalization)
# --------------------------------------------------------------------------------------
SCHEME_RE = re.compile(r'^[a-z][a-z0-9+.\-]*://', re.IGNORECASE)
WWW_RE = re.compile(r'^www\d{0,3}\.')
AUTHORITY_RE = re.compile(r'^([^/?#]*)(.*)$', re.DOTALL)   # authority | phần còn lại


def normalize_url(url) -> str:
    """Bỏ scheme, viết thường host, bỏ `www.`, bỏ `.` cuối host và `/` cuối URL."""
    u = SCHEME_RE.sub('', str(url).strip())
    authority, rest = AUTHORITY_RE.match(u).groups()
    authority = WWW_RE.sub('', authority.lower()).rstrip('.')
    return authority + rest.rstrip('/')


TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())   # offline, tái lập được


def split_components(url_norm: str):
    """URL chuẩn hóa → (host, has_port, has_userinfo, path, query)."""
    try:
        parts = urlsplit('http://' + url_norm)
        has_port = parts.port is not None            # có thể ném ValueError nếu port sai định dạng
        return (parts.hostname or ''), int(has_port), int('@' in parts.netloc), parts.path, parts.query
    except ValueError:                               # URL dị dạng → tách thủ công
        authority, rest = AUTHORITY_RE.match(url_norm).groups()
        hostport = authority.rsplit('@', 1)[-1]
        path, _, query = rest.partition('?')
        return hostport.split(':')[0], int(':' in hostport), int('@' in authority), path, query


def host_parts(host: str):
    """host → (subdomain, sld, suffix, registered_domain) theo Public Suffix List."""
    r = TLD_EXTRACT(host)
    registered = (r.top_domain_under_public_suffix if hasattr(r, 'top_domain_under_public_suffix')
                  else r.registered_domain)
    return r.subdomain, r.domain, r.suffix, (registered or host)


def registered_domain(url) -> str:
    return host_parts(split_components(normalize_url(url))[0])[3]


# --------------------------------------------------------------------------------------
# 2. Từ điển tri thức bảo mật
# --------------------------------------------------------------------------------------
BRANDS = ['paypal', 'apple', 'icloud', 'itunes', 'microsoft', 'office365', 'outlook', 'hotmail',
          'google', 'gmail', 'facebook', 'instagram', 'whatsapp', 'amazon', 'netflix', 'ebay',
          'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'hsbc', 'barclays', 'santander',
          'dropbox', 'docusign', 'adobe', 'yahoo', 'linkedin', 'twitter', 'usaa', 'americanexpress',
          'amex', 'alibaba', 'fedex', 'mastercard', 'coinbase', 'binance', 'blockchain', 'bitcoin']
BRAND_SET = set(BRANDS)
TYPO_BRANDS = [b for b in BRANDS if len(b) >= 5]
BRAND_RE = re.compile(r'(?<![a-z])(' + '|'.join(sorted(BRANDS, key=len, reverse=True)) + ')')

SENSITIVE_WORDS = ['login', 'log-in', 'logon', 'signin', 'sign-in', 'verify', 'verification', 'validate',
                   'account', 'update', 'confirm', 'secure', 'security', 'banking', 'password', 'passwd',
                   'webscr', 'cmd', 'ebayisapi', 'wallet', 'suspend', 'suspended', 'unlock', 'billing',
                   'invoice', 'recover', 'recovery', 'authenticate', 'authorize', 'credential', 'restore']
SENSITIVE_RE = re.compile(r'(?<![a-z])(?:' + '|'.join(map(re.escape, SENSITIVE_WORDS)) + r')(?![a-z])')

SHORTENERS = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do',
              'cutt.ly', 'shorturl.at', 'rebrand.ly', 'tiny.cc', 'rb.gy', 'x.co', 'lnkd.in', 'db.tt',
              'qr.net', '1url.com', 'tr.im', 'v.gd', 'tinyurl.co', 'shorte.st', 'soo.gd'}

EXT_GROUPS = {
    'ext_executable': {'exe', 'scr', 'bat', 'msi', 'apk', 'jar', 'dll', 'bin', 'vbs', 'ps1', 'cmd'},
    'ext_script':     {'php', 'asp', 'aspx', 'jsp', 'cgi', 'pl', 'py'},
    'ext_html':       {'html', 'htm', 'shtml', 'xhtml'},
    'ext_archive':    {'zip', 'rar', '7z', 'gz', 'tar', 'tgz'},
    'ext_document':   {'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'rtf'},
}

TOKEN_SPLIT_RE = re.compile(r'[^a-z0-9]+')
EXT_RE = re.compile(r'[a-z0-9]{1,5}')
EMBED_RE = re.compile(r'(?:https?:|https?%3a|www\.)')
REDIRECT_RE = re.compile(r'(?:^|[&;])(?:url|redirect|redir|next|goto|return|returnurl|dest|destination|'
                         r'continue|target|link|out)=')
IP_LIKE_RE = re.compile(r'(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+)){0,3}')
VOWELS = set('aeiou')


# --------------------------------------------------------------------------------------
# 3. Hàm tiện ích
# --------------------------------------------------------------------------------------
def shannon_entropy(s: str) -> float:
    """Entropy Shannon (bit/ký tự)."""
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def vowel_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(c in VOWELS for c in letters) / len(letters) if letters else 0.0


def max_consonant_run(s: str) -> int:
    best = run = 0
    for c in s:
        run = run + 1 if (c.isalpha() and c not in VOWELS) else 0
        best = max(best, run)
    return best


def bounded_levenshtein(a: str, b: str, max_d: int = 3) -> int:
    """Levenshtein có chặn trên max_d (dừng sớm)."""
    if abs(len(a) - len(b)) >= max_d:
        return max_d
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) >= max_d:
            return max_d
        prev = cur
    return min(prev[-1], max_d)


@lru_cache(maxsize=None)
def typosquat_info(sld: str):
    """(khoảng cách nhỏ nhất tới brand ∈ [0..3], cờ typosquat)."""
    if len(sld) < 4:
        return 3, 0
    best, flag = 3, 0
    for b in TYPO_BRANDS:
        d = bounded_levenshtein(sld, b)
        best = min(best, d)
        if 1 <= d <= (1 if len(b) <= 6 else 2):
            flag = 1
    return best, flag


def is_ip_host(host: str) -> bool:
    """IPv4/IPv6 chuẩn + dạng né tránh (số nguyên, hex, octal)."""
    if not host:
        return False
    h = host.strip('[]')
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return bool(IP_LIKE_RE.fullmatch(h))


# --------------------------------------------------------------------------------------
# 4. Trích xuất đặc trưng
# --------------------------------------------------------------------------------------
def host_level_features(host: str) -> dict:
    sub, sld, suffix, registered = host_parts(host)
    ip = is_ip_host(host)
    if ip:
        sub, sld, suffix = '', '', ''
    typo_dist, is_typo = typosquat_info(sld)
    sld_alnum = sum(c.isalnum() for c in sld)
    return {
        'host_len': len(host),
        'host_n_dots': host.count('.'),
        'host_n_hyphens': host.count('-'),
        'host_n_digits': sum(c.isdigit() for c in host),
        'host_entropy': shannon_entropy(host),
        'host_is_ip': int(ip),
        'host_has_punycode': int('xn--' in host),
        'host_non_ascii': int(any(ord(c) > 127 for c in host)),
        'subdomain_levels': len(sub.split('.')) if sub else 0,
        'subdomain_len': len(sub),
        'sld_len': len(sld),
        'sld_entropy': shannon_entropy(sld),
        'sld_digit_ratio': sum(c.isdigit() for c in sld) / sld_alnum if sld_alnum else 0.0,
        'sld_vowel_ratio': vowel_ratio(sld),
        'sld_max_consonant_run': max_consonant_run(sld),
        'sld_n_hyphens': sld.count('-'),
        'tld_len': len(suffix),
        'is_shortener': int(registered in SHORTENERS),
        'sld_is_brand': int(sld in BRAND_SET),
        'brand_combo_sld': int(sld not in BRAND_SET and bool(BRAND_RE.search(sld))),
        'typosquat_dist': typo_dist,
        'is_typosquat': is_typo,
        'tld': '<ip>' if ip else (suffix or '<none>'),
        '_sub': sub, '_sld': sld,
    }


def row_level_features(url_norm, host, path, query, has_port, has_userinfo, h) -> dict:
    low = url_norm.lower()
    pq = (path + ('?' + query if query else '')).lower()
    n = max(len(url_norm), 1)
    letters = sum(c.isalpha() for c in url_norm)
    digits = sum(c.isdigit() for c in url_norm)
    tokens = [t for t in TOKEN_SPLIT_RE.split(low) if t]
    segments = [s for s in path.split('/') if s]
    last = segments[-1].lower() if segments else ''
    ext = last.rsplit('.', 1)[1] if '.' in last else ''
    ext = ext if EXT_RE.fullmatch(ext) else ''
    brands_outside = set(BRAND_RE.findall(h['_sub'])) | set(BRAND_RE.findall(pq))

    feats = {
        'url_len': len(url_norm),
        'path_len': len(path),
        'query_len': len(query),
        'path_depth': len(segments),
        'n_params': (query.count('&') + 1) if query else 0,
        'n_dots': url_norm.count('.'),
        'n_hyphens': url_norm.count('-'),
        'n_underscores': url_norm.count('_'),
        'n_slashes': url_norm.count('/'),
        'n_at': url_norm.count('@'),
        'n_tilde': url_norm.count('~'),
        'n_percent': url_norm.count('%'),
        'n_equals': url_norm.count('='),
        'digit_ratio': digits / n,
        'letter_ratio': letters / n,
        'special_ratio': (n - letters - digits) / n,
        'upper_ratio': sum(c.isupper() for c in url_norm) / max(letters, 1),
        'n_tokens': len(tokens),
        'longest_token': max(map(len, tokens), default=0),
        'avg_token_len': (sum(map(len, tokens)) / len(tokens)) if tokens else 0.0,
        'url_entropy': shannon_entropy(low),
        'path_entropy': shannon_entropy(pq),
        'has_port': has_port,
        'has_userinfo': has_userinfo,
        'double_slash_in_path': int('//' in path),
        'has_embedded_url': int(bool(EMBED_RE.search(pq))),
        'has_redirect_param': int(bool(REDIRECT_RE.search(query.lower()))),
        'has_file_ext': int(bool(ext)),
        **{group: int(ext in exts) for group, exts in EXT_GROUPS.items()},
        'sens_in_host': len(SENSITIVE_RE.findall(host)),
        'sens_in_path': len(SENSITIVE_RE.findall(pq)),
        'brand_in_subdomain': int(bool(BRAND_RE.search(h['_sub']))),
        'brand_in_path': int(bool(BRAND_RE.search(pq))),
        'brand_out_of_domain': int(any(b not in h['_sld'] for b in brands_outside)),
    }
    feats.update({k: v for k, v in h.items() if not k.startswith('_')})
    return feats


def build_feature_frame(raw_urls, chunk_size: int = 50_000) -> pd.DataFrame:
    """URL thô → DataFrame đặc trưng (60 cột số + 1 cột phân loại `tld`). Dùng chung train/inference."""
    raw_urls = list(raw_urls)
    host_cache, frames = {}, []
    for start in range(0, len(raw_urls), chunk_size):
        rows = []
        for u in raw_urls[start:start + chunk_size]:
            nu = normalize_url(u)
            host, has_port, has_ui, path, query = split_components(nu)
            h = host_cache.get(host)
            if h is None:
                h = host_cache[host] = host_level_features(host)
            rows.append(row_level_features(nu, host, path, query, has_port, has_ui, h))
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


# Registry tường minh: đặc trưng → (nhóm, kỹ thuật tấn công mục tiêu)
FEATURE_GROUPS = {
    'Cấu trúc URL': (['url_len', 'path_len', 'query_len', 'path_depth', 'n_params', 'n_dots', 'n_hyphens',
                      'n_underscores', 'n_slashes', 'n_at', 'n_tilde', 'n_percent', 'n_equals'],
                     'Hình thái tổng quát, URL nhồi tham số'),
    'Thành phần ký tự': (['digit_ratio', 'letter_ratio', 'special_ratio', 'upper_ratio', 'n_tokens',
                          'longest_token', 'avg_token_len'], 'Chuỗi hash MD5/SHA, payload mã hóa'),
    'Tính ngẫu nhiên (DGA)': (['host_entropy', 'sld_entropy', 'url_entropy', 'path_entropy', 'sld_vowel_ratio',
                               'sld_max_consonant_run', 'sld_digit_ratio'], 'Domain sinh tự động (DGA)'),
    'Host / Domain': (['host_len', 'host_n_dots', 'host_n_hyphens', 'host_n_digits', 'subdomain_levels',
                       'subdomain_len', 'sld_len', 'sld_n_hyphens', 'tld_len'], 'Subdomain giả, hạ tầng giá rẻ'),
    'Thương hiệu': (['brand_out_of_domain', 'brand_in_subdomain', 'brand_in_path', 'brand_combo_sld',
                     'sld_is_brand'], 'Brand impersonation, combosquatting'),
    'Typosquatting': (['typosquat_dist', 'is_typosquat'], 'paypa1.com, gooogle.com'),
    'Từ khóa theo vị trí': (['sens_in_host', 'sens_in_path'], 'Trang đăng nhập giả'),
    'Né tránh & payload': (['host_is_ip', 'host_has_punycode', 'host_non_ascii', 'has_userinfo', 'has_port',
                            'is_shortener', 'double_slash_in_path', 'has_embedded_url', 'has_redirect_param',
                            'has_file_ext', 'ext_executable', 'ext_script', 'ext_html', 'ext_archive',
                            'ext_document'], 'Che giấu đích đến, open redirect, phát tán malware'),
    'TLD (phân loại)': (['tld'], 'TLD rẻ / ít kiểm duyệt'),
}
FEATURE_TO_GROUP = {f: g for g, (fs, _) in FEATURE_GROUPS.items() for f in fs}


# --------------------------------------------------------------------------------------
# 5. Đặc trưng CŨ (20 cột của notebook 04/05) – chỉ để tái lập baseline
# --------------------------------------------------------------------------------------
LEGACY_COLS = ['get_url_length', 'get_hostname_length', 'count_dots', 'count_hyphens', 'count_at_symbol',
               'count_slashes', 'count_question_mark', 'count_equals', 'count_and_symbol', 'get_path_length',
               'count_digits_in_url', 'count_letters_in_url', 'count_digits_in_hostname', 'has_https',
               'has_suspicious_tld', 'count_sensitive_words', 'has_ip_address_in_hostname',
               'get_subdomain_count', 'get_domain_age_days', 'get_domain_lifespan_days']


def legacy_feature_frame(raw_urls) -> pd.DataFrame:
    """Gọi NGUYÊN VẸN các hàm trong src/features (không sửa code cũ), đúng thứ tự cột X_train của NB05."""
    lex = importlib.import_module('src.features.lexical_features')
    host = importlib.import_module('src.features.host_features')
    funcs = {c: getattr(lex, c, None) or getattr(host, c) for c in LEGACY_COLS}
    raw_urls = list(raw_urls)
    return pd.DataFrame({c: [f(u) for u in raw_urls] for c, f in funcs.items()}).astype(float)
