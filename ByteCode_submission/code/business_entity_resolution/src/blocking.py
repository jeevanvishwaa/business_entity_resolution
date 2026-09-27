"""
High-performance candidate generation (blocking) engine.
Provides canonical blocking semantics and shared representations across both
in-memory training (CandidateIndex) and disk-backed SQLite test inference (SQLiteCandidateLookup).
"""

import os
import sqlite3
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

FLAG_NAME_EXACT = 1
FLAG_NAME_COMPACT = 2
FLAG_NAME_PREFIX = 4
FLAG_NAME_TOKEN = 8
FLAG_ADDR_POSTAL = 16
FLAG_ADDR_STREET = 32
FLAG_ADDR_COMPACT = 64

PROV_NAMES = {
    FLAG_NAME_EXACT: 'block_name_exact',
    FLAG_NAME_COMPACT: 'block_name_compact',
    FLAG_NAME_PREFIX: 'block_name_prefix',
    FLAG_NAME_TOKEN: 'block_name_token',
    FLAG_ADDR_POSTAL: 'block_addr_postal',
    FLAG_ADDR_STREET: 'block_addr_street',
    FLAG_ADDR_COMPACT: 'block_addr_compact'
}

def extract_address_blocking_key(norm_addr: dict) -> Tuple[Optional[str], int]:
    """
    Canonical helper to extract address blocking key and corresponding bitwise flag.
    Returns (key, flag) or (None, 0).
    """
    s_num = norm_addr.get('street_number', '')
    p_code = norm_addr.get('postal_code', '')
    if s_num and p_code:
        return f"{s_num}_{p_code}", FLAG_ADDR_POSTAL
    tokens = norm_addr.get('tokens', [])
    if s_num and len(tokens) >= 3:
        for t in tokens[1:3]:
            if len(t) >= 4 and not t.isdigit():
                return f"{s_num}_{t}", FLAG_ADDR_STREET
    return None, 0

def rank_and_truncate_candidates(cand_flags: Dict[str, int], max_candidates: int) -> List[Tuple[str, List[str]]]:
    """
    Canonical candidate ranking and truncation.
    Ranks candidates by popcount (number of matching blocking strategies) descending,
    truncates to max_candidates, and extracts the list of provenance block names.
    Returns: List of (candidate_id, [provenance_block_names])
    """
    if not cand_flags:
        return []
    # Sort by number of matching blocks (popcount) descending, tie-breaking by candidate ID for determinism
    ranked = sorted(cand_flags.items(), key=lambda item: (-bin(item[1]).count('1'), item[0]))
    if len(ranked) > max_candidates:
        ranked = ranked[:max_candidates]
    return [
        (cid, [name for flag, name in PROV_NAMES.items() if (mask & flag)])
        for cid, mask in ranked
    ]

class CompactRecord:
    __slots__ = (
        'raw_name', 'compact_name', 'tokens_name', 'core_tokens_name',
        'raw_addr', 'tokens_addr', 'numbers_addr', 'postal_code'
    )
    def __init__(self, n_dict: dict, a_dict: dict):
        self.raw_name = n_dict['raw_clean']
        self.compact_name = n_dict['compact']
        self.tokens_name = n_dict['tokens']
        self.core_tokens_name = n_dict['core_tokens']
        self.raw_addr = a_dict['raw_clean']
        self.tokens_addr = a_dict['tokens']
        self.numbers_addr = a_dict['numbers']
        self.postal_code = a_dict['postal_code']

class CandidateIndex:
    """
    In-memory candidate generation engine used during training and validation pair preparation.
    Uses lightweight __slots__ records and integer posting lists.
    """
    def __init__(self, country: str, max_post_len: int = 400):
        self.country = country
        self.max_post_len = max_post_len

        # Compact storage
        self.id_list = []          # idx -> entity_id (str)
        self.id_to_idx = {}        # entity_id -> idx (int)
        self.records = []          # idx -> CompactRecord

        # Inverted indexes mapping key -> list of integer indices (idx)
        self.name_exact = defaultdict(list)
        self.name_compact = defaultdict(list)
        self.name_prefix = defaultdict(list)
        self.name_distinctive = defaultdict(list)
        self.addr_num_postal = defaultdict(list)
        self.addr_compact = defaultdict(list)

        self.token_freq = defaultdict(int)

    def get_record(self, entity_id: str):
        """O(1) retrieval of record by entity_id."""
        idx = self.id_to_idx.get(entity_id)
        return self.records[idx] if idx is not None else None

    def add_target_record(self, entity_id: str, norm_name: dict, norm_addr: dict):
        """Index a Source 2 or Source 3 record using integer references."""
        idx = len(self.id_list)
        self.id_list.append(entity_id)
        self.id_to_idx[entity_id] = idx
        self.records.append(CompactRecord(norm_name, norm_addr))

        # 1. Exact clean name
        if norm_name['raw_clean']:
            self.name_exact[norm_name['raw_clean']].append(idx)

        # 2. Compact name
        if norm_name['compact'] and len(norm_name['compact']) >= 4:
            self.name_compact[norm_name['compact']].append(idx)

        # 3. Prefix key
        if norm_name['prefix_key'] and len(norm_name['prefix_key']) >= 4:
            self.name_prefix[norm_name['prefix_key']].append(idx)

        # 4. Distinctive tokens
        for tok in norm_name['distinctive_tokens']:
            self.token_freq[tok] += 1
            if self.token_freq[tok] <= self.max_post_len:
                self.name_distinctive[tok].append(idx)

        # 5. Address number + postal/locality
        addr_key, _ = extract_address_blocking_key(norm_addr)
        if addr_key and len(self.addr_num_postal[addr_key]) <= self.max_post_len:
            self.addr_num_postal[addr_key].append(idx)

        # 6. Compact address
        if norm_addr['compact'] and len(norm_addr['compact']) >= 10:
            if len(self.addr_compact[norm_addr['compact']]) <= self.max_post_len:
                self.addr_compact[norm_addr['compact']].append(idx)

    def prune_high_frequency_tokens(self):
        """Prune overly frequent posting lists to avoid memory blowup on generic terms."""
        for tok, count in list(self.token_freq.items()):
            if count > self.max_post_len and tok in self.name_distinctive:
                del self.name_distinctive[tok]

        for key in list(self.name_prefix.keys()):
            if len(self.name_prefix[key]) > self.max_post_len:
                del self.name_prefix[key]

    def retrieve_candidates(self, norm_name: dict, norm_addr: dict, max_candidates: int = 50) -> List[Tuple[str, List[str]]]:
        """
        Retrieves candidate entity_ids for a query S1 entity using integer flags.
        Returns list of (candidate_id, [provenance_blocks]).
        """
        cand_flags = defaultdict(int)

        # Block 1: Exact Clean Name
        if norm_name['raw_clean']:
            for idx in self.name_exact.get(norm_name['raw_clean'], []):
                cand_flags[idx] |= FLAG_NAME_EXACT

        # Block 2: Compact Name
        if norm_name['compact'] and len(norm_name['compact']) >= 4:
            for idx in self.name_compact.get(norm_name['compact'], []):
                cand_flags[idx] |= FLAG_NAME_COMPACT

        # Block 3: Prefix Name
        if norm_name['prefix_key']:
            posting = self.name_prefix.get(norm_name['prefix_key'], [])
            if len(posting) <= self.max_post_len:
                for idx in posting:
                    cand_flags[idx] |= FLAG_NAME_PREFIX

        # Block 4: Distinctive Name Tokens
        for tok in norm_name['distinctive_tokens']:
            posting = self.name_distinctive.get(tok, [])
            if 0 < len(posting) <= self.max_post_len:
                for idx in posting:
                    cand_flags[idx] |= FLAG_NAME_TOKEN

        # Block 5: Address number + postal/street token
        addr_key, addr_flag = extract_address_blocking_key(norm_addr)
        if addr_key:
            posting = self.addr_num_postal.get(addr_key, [])
            if len(posting) <= self.max_post_len:
                for idx in posting:
                    cand_flags[idx] |= addr_flag

        # Block 6: Compact Address
        if norm_addr['compact'] and len(norm_addr['compact']) >= 10:
            for idx in self.addr_compact.get(norm_addr['compact'], []):
                cand_flags[idx] |= FLAG_ADDR_COMPACT

        if not cand_flags:
            return []

        # Convert idx keys to entity_id keys and rank
        cand_flags_by_id = {self.id_list[idx]: mask for idx, mask in cand_flags.items()}
        return rank_and_truncate_candidates(cand_flags_by_id, max_candidates=max_candidates)

class SQLiteCandidateLookup:
    """
    Disk-backed, SQLite-based candidate generator implementing the canonical
    blocking semantics for test inference under strict memory constraints (< 200 MB RAM).
    """
    def __init__(self, db_path: str, max_post_len: int = 400):
        self.db_path = db_path
        self.max_post_len = max_post_len
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self._setup_pragmas()
        self._cache = {}

    def _setup_pragmas(self):
        self.cur.execute("PRAGMA synchronous = OFF")
        self.cur.execute("PRAGMA journal_mode = MEMORY")
        self.cur.execute("PRAGMA cache_size = -100000")  # 100 MB cache
        self.cur.execute("PRAGMA temp_store = MEMORY")

    @classmethod
    def init_database(cls, db_path: str, max_post_len: int = 400):
        """Initializes tables and returns an active lookup instance."""
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA synchronous = OFF")
        cur.execute("PRAGMA journal_mode = MEMORY")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS targets (
            id TEXT PRIMARY KEY,
            name TEXT,
            compact TEXT,
            prefix TEXT,
            addr_key TEXT,
            addr_compact TEXT,
            raw_addr TEXT,
            raw_name TEXT,
            num TEXT,
            postal TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS target_tokens (
            tok TEXT,
            id TEXT
        )
        """)
        conn.commit()
        conn.close()
        return cls(db_path, max_post_len=max_post_len)

    def add_target_batch(self, batch_records: List[Tuple[str, dict, dict]], commit: bool = True):
        """
        Inserts a batch of normalized target records into SQLite.
        batch_records: List of (cid, norm_name, norm_addr)
        """
        target_rows = []
        token_rows = []

        for cid, n_dict, a_dict in batch_records:
            addr_key, _ = extract_address_blocking_key(a_dict)
            compact_name = n_dict['compact'] if len(n_dict['compact']) >= 4 else ''
            prefix_key = n_dict['prefix_key'] if len(n_dict['prefix_key']) >= 4 else ''
            addr_compact = a_dict['compact'] if len(a_dict['compact']) >= 10 else ''

            target_rows.append((
                cid,
                n_dict['raw_clean'],
                compact_name,
                prefix_key,
                addr_key or '',
                addr_compact,
                a_dict['raw_clean'],
                n_dict['raw_clean'],
                a_dict['street_number'],
                a_dict['postal_code']
            ))

            for tok in n_dict['distinctive_tokens']:
                token_rows.append((tok, cid))

        if target_rows:
            self.cur.executemany("INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", target_rows)
        if token_rows:
            self.cur.executemany("INSERT INTO target_tokens VALUES (?, ?)", token_rows)
        if commit:
            self.conn.commit()

    def finalize_indexes(self):
        """Builds B-tree indexes and prunes overly frequent posting lists."""
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_name ON targets (name)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_compact ON targets (compact)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_prefix ON targets (prefix)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_addr_key ON targets (addr_key)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_addr_compact ON targets (addr_compact)")
        self.cur.execute("CREATE INDEX IF NOT EXISTS idx_tok ON target_tokens (tok)")
        
        # Prune high-frequency tokens
        self.cur.execute("""
        DELETE FROM target_tokens WHERE tok IN (
            SELECT tok FROM target_tokens GROUP BY tok HAVING count(*) > ?
        )
        """, (self.max_post_len,))
        
        # Prune high-frequency prefixes
        self.cur.execute("""
        UPDATE targets SET prefix = '' WHERE prefix IN (
            SELECT prefix FROM targets WHERE prefix != '' GROUP BY prefix HAVING count(*) > ?
        )
        """, (self.max_post_len,))

        # Prune high-frequency address keys
        self.cur.execute("""
        UPDATE targets SET addr_key = '' WHERE addr_key IN (
            SELECT addr_key FROM targets WHERE addr_key != '' GROUP BY addr_key HAVING count(*) > ?
        )
        """, (self.max_post_len,))
        self.conn.commit()

    def retrieve_candidates(self, norm_name: dict, norm_addr: dict, max_candidates: int = 50) -> List[Tuple[str, List[str]]]:
        """
        Retrieves candidates for a query S1 entity using canonical blocking semantics.
        Returns: List of (candidate_id, [provenance_blocks])
        """
        cand_flags = defaultdict(int)

        q_name = norm_name.get('raw_clean', '')
        q_compact = norm_name.get('compact', '') if len(norm_name.get('compact', '')) >= 4 else ''
        q_prefix = norm_name.get('prefix_key', '')
        q_addr_key, q_addr_flag = extract_address_blocking_key(norm_addr)
        q_addr_compact = norm_addr.get('compact', '') if len(norm_addr.get('compact', '')) >= 10 else ''
        q_tokens = norm_name.get('distinctive_tokens', [])

        has_target_queries = any([q_name, q_compact, q_prefix, q_addr_key, q_addr_compact])

        if has_target_queries:
            query_sql = """
            SELECT id, raw_name, compact, raw_addr, num, postal,
                (
                    (CASE WHEN ? != '' AND name != '' AND name = ? THEN 1 ELSE 0 END) |
                    (CASE WHEN ? != '' AND compact != '' AND compact = ? THEN 2 ELSE 0 END) |
                    (CASE WHEN ? != '' AND prefix != '' AND prefix = ? THEN 4 ELSE 0 END) |
                    (CASE WHEN ? != '' AND addr_key != '' AND addr_key = ? THEN ? ELSE 0 END) |
                    (CASE WHEN ? != '' AND addr_compact != '' AND addr_compact = ? THEN 64 ELSE 0 END)
                ) AS prov_mask
            FROM targets
            WHERE (? != '' AND name != '' AND name = ?)
               OR (? != '' AND compact != '' AND compact = ?)
               OR (? != '' AND prefix != '' AND prefix = ?)
               OR (? != '' AND addr_key != '' AND addr_key = ?)
               OR (? != '' AND addr_compact != '' AND addr_compact = ?)
            """
            params = (
                q_name, q_name,
                q_compact, q_compact,
                q_prefix, q_prefix,
                q_addr_key or '', q_addr_key or '', q_addr_flag,
                q_addr_compact, q_addr_compact,
                q_name, q_name,
                q_compact, q_compact,
                q_prefix, q_prefix,
                q_addr_key or '', q_addr_key or '',
                q_addr_compact, q_addr_compact
            )
            self.cur.execute(query_sql, params)
            for row in self.cur.fetchall():
                cid, r_name, t_cmp, r_addr, num, postal, mask = row
                if mask > 0:
                    cand_flags[cid] |= mask
                    self._cache[cid] = (r_name, t_cmp, r_addr, num, postal)

        # Distinctive tokens block
        for tok in q_tokens:
            self.cur.execute("SELECT id FROM target_tokens WHERE tok = ? LIMIT ?", (tok, self.max_post_len + 1))
            tok_rows = self.cur.fetchall()
            if 0 < len(tok_rows) <= self.max_post_len:
                for r in tok_rows:
                    cid = r[0]
                    cand_flags[cid] |= FLAG_NAME_TOKEN

        return rank_and_truncate_candidates(cand_flags, max_candidates=max_candidates)

    def get_record(self, entity_id: str) -> Optional[dict]:
        """
        Retrieves normalized target record formatted for compute_pairwise_features.
        Uses in-memory query cache when available, falling back to SQLite query.
        """
        if entity_id in self._cache:
            r_name, t_cmp, r_addr, num, postal = self._cache[entity_id]
        else:
            self.cur.execute("SELECT raw_name, compact, raw_addr, num, postal FROM targets WHERE id = ?", (entity_id,))
            row = self.cur.fetchone()
            if not row:
                return None
            r_name, t_cmp, r_addr, num, postal = row
            self._cache[entity_id] = row

        return {
            'norm_name': {
                'raw_clean': r_name or '',
                'compact': t_cmp or '',
                'tokens': (r_name or '').split(),
                'core_tokens': (r_name or '').split()
            },
            'norm_addr': {
                'raw_clean': r_addr or '',
                'tokens': (r_addr or '').split(),
                'numbers': [num] if num else [],
                'postal_code': postal or ''
            }
        }

    def get_records_batch(self, entity_ids: List[str]) -> Dict[str, dict]:
        """
        Retrieves multiple normalized target records using a single batched SQLite query.
        Significantly reduces query roundtrip overhead during candidate feature generation.
        """
        missing = [cid for cid in entity_ids if cid not in self._cache]
        if missing:
            placeholders = ','.join('?' * len(missing))
            self.cur.execute(
                f"SELECT id, raw_name, compact, raw_addr, num, postal FROM targets WHERE id IN ({placeholders})",
                missing
            )
            for row in self.cur.fetchall():
                cid, r_name, t_cmp, r_addr, num, postal = row
                self._cache[cid] = (r_name, t_cmp, r_addr, num, postal)
        return {cid: self.get_record(cid) for cid in entity_ids}

    def clear_cache(self):
        """Clears record cache to maintain strictly bounded memory."""
        self._cache.clear()

    def close(self):
        """Closes the underlying SQLite connection."""
        self.conn.close()
