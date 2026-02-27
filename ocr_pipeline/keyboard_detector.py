import re
from typing import List, Tuple, Dict

# Standard QWERTY keyboard layout rows
QWERTY_ROWS = [
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm']
]

# Valid 3-char horizontal sequences from any QWERTY row
VALID_SEQUENCES = {
    row_str[i:i+3]
    for row in QWERTY_ROWS
    for row_str in ["".join(row)]
    for i in range(len(row_str) - 2)
}

# All keyboard letters + digits
KEYBOARD_LETTERS = {c for row in QWERTY_ROWS for c in row} | set("0123456789")

# Keyboard UI element keywords
KEYBOARD_UI_ELEMENTS = {
    'space', 'return', 'enter', 'shift', 'delete', 'backspace',
    'go', 'send', 'search', 'next', 'done', 'abc', '123',
    '@', '#+=', '.?123', 'emoji', '?123', 'english',
}

# Default detection parameters
SCAN_REGION_FRACTION = 0.70   # Bottom fraction of image to scan
MIN_KEYBOARD_ROWS    = 2      # Minimum distinct rows to confirm keyboard
MIN_CHARS_PER_ROW    = 5      # Minimum key-like characters per row
ROW_CLUSTER_THRESHOLD = 60    # Y-pixel tolerance for grouping into a row


def _center_y(item: dict) -> float:
    box = item['box']
    return (box[0][1] + box[2][1]) / 2


class ImprovedKeyboardDetector:
    """
    Detects on-screen keyboards in portrait screenshots using spatial
    row-clustering and position-based validation.
    """

    def __init__(self,
                 scan_fraction: float = SCAN_REGION_FRACTION,
                 min_rows: int = MIN_KEYBOARD_ROWS,
                 min_chars_per_row: int = MIN_CHARS_PER_ROW,
                 row_threshold: int = ROW_CLUSTER_THRESHOLD):
        self.scan_fraction     = scan_fraction
        self.min_rows          = min_rows
        self.min_chars_per_row = min_chars_per_row
        self.row_threshold     = row_threshold

    # Text classification helpers

    def _is_keyboard_key(self, text: str) -> bool:
        c = text.strip().lower()
        return len(c) == 1 and c in KEYBOARD_LETTERS

    def _is_ui_element(self, text: str) -> bool:
        return text.strip().lower() in KEYBOARD_UI_ELEMENTS

    def _key_sequence_length(self, text: str) -> int:
        """
        Return len(text) if it looks like grouped OCR'd keyboard keys, else 0.

        TWO guards prevent false positives on natural-language text:
        1. Max length: real grouped key blocks are short (e.g. 'qwerty' = 6 chars).
        2. Purity: every character must be a keyboard letter/digit.
        """
        cleaned = text.strip().lower().replace(" ", "")
        # Guard 1 – too long to be a grouped key block
        if len(cleaned) < 3 or len(cleaned) > 10:
            return 0
        # Guard 2 – must be entirely keyboard characters (letters + digits)
        if not all(c in KEYBOARD_LETTERS for c in cleaned):
            return 0
        for i in range(len(cleaned) - 2):
            if cleaned[i:i+3] in VALID_SEQUENCES:
                return len(cleaned)
        return 0

    # ------------------------------------------------------------------
    # Row discovery
    # ------------------------------------------------------------------

    def _cluster_by_y(self, items: List[Dict], threshold: int) -> List[List[Dict]]:
        """Group items into horizontal rows by proximity of their Y-centers."""
        if not items:
            return []
        sorted_items = sorted(items, key=lambda x: x['y_pos'])
        clusters, current = [], [sorted_items[0]]
        for item in sorted_items[1:]:
            median_y = sorted(i['y_pos'] for i in current)[len(current) // 2]
            if abs(item['y_pos'] - median_y) <= threshold:
                current.append(item)
            else:
                clusters.append(current)
                current = [item]
        clusters.append(current)
        return clusters

    def find_character_rows(self, ocr_data: List[dict], image_height: int) -> List[Dict]:
        """Find horizontal rows of keyboard-like characters in the scan region."""
        bottom_start = image_height * (1 - self.scan_fraction)
        candidates = []

        for item in ocr_data:
            box  = item['box']
            y_top, y_bot = box[0][1], box[2][1]
            cy   = (y_top + y_bot) / 2
            if cy < bottom_start:
                continue

            text = item['text'].strip()
            is_key = self._is_keyboard_key(text)
            is_ui  = self._is_ui_element(text)

            if is_key or is_ui or len(text) <= 2:
                candidates.append({
                    'text': text, 'y_pos': cy,
                    'y_top': y_top, 'y_bottom': y_bot, 'box': box,
                    'is_key': is_key, 'is_ui': is_ui,
                    'key_count': 1 if is_key else 0
                })
            else:
                seq_len = self._key_sequence_length(text)
                if seq_len > 0:
                    candidates.append({
                        'text': text, 'y_pos': cy,
                        'y_top': y_top, 'y_bottom': y_bot, 'box': box,
                        'is_key': True, 'is_ui': False,
                        'key_count': seq_len
                    })

        rows = []
        for cluster in self._cluster_by_y(candidates, self.row_threshold):
            key_count = sum(i['key_count'] for i in cluster)
            ui_count  = sum(1 for i in cluster if i['is_ui'])
            rows.append({
                'y_start':   min(i['y_top']    for i in cluster),
                'y_end':     max(i['y_bottom'] for i in cluster),
                'y_center':  sum(i['y_pos']    for i in cluster) / len(cluster),
                'char_count': key_count + ui_count,
                'key_count':  key_count,
                'ui_count':   ui_count,
                'items':      cluster,
            })
        return rows

    # ------------------------------------------------------------------
    # Strong-UI-row check
    # ------------------------------------------------------------------

    def _is_strong_ui_row(self, row: Dict) -> bool:
        """True if the row contains combinations that strongly imply a keyboard."""
        texts = {i['text'].lower() for i in row['items']}
        has_english = 'english' in texts
        has_space   = 'space'   in texts
        has_numbers = bool(texts & {'?123', '.?123', '123'})
        has_enter   = bool(texts & {'return', 'enter', 'go', 'search'})
        return (
            (has_english and has_numbers) or
            (has_english and has_enter)   or
            (has_space   and has_enter)   or
            (has_space   and has_numbers)
        )

    # ------------------------------------------------------------------
    # Region detection (main public API)
    # ------------------------------------------------------------------

    def detect_keyboard_regions(self, ocr_data: List[dict], image_height: int) -> List[Tuple[float, float]]:
        """
        Return a list of (start_frac, end_frac) tuples marking keyboard regions.
        All fractions are relative to image_height.
        """
        rows = self.find_character_rows(ocr_data, image_height)
        if not rows:
            return []

        # Step 1 – Anchor rows: strong UI or sufficient character density
        anchor_rows = [r for r in rows
                       if self._is_strong_ui_row(r) or r['char_count'] >= self.min_chars_per_row]
        if not anchor_rows:
            return []

        # Step 2 – Promote adjacent weak rows that contain at least one key/UI token
        proximity = image_height * 0.15
        valid_set = set(id(r) for r in anchor_rows)
        valid_rows = list(anchor_rows)
        for row in rows:
            if id(row) in valid_set:
                continue
            near_anchor = any(abs(row['y_center'] - a['y_center']) <= proximity
                              for a in anchor_rows)
            # Require ≥2 key items before promoting a weak row.
            if near_anchor and (row['key_count'] >= 2 or row['ui_count'] > 0):
                valid_rows.append(row)
                valid_set.add(id(row))

        # Step 3 – Cluster valid rows into contiguous groups
        sorted_rows = sorted(valid_rows, key=lambda r: r['y_center'])
        spacing     = image_height * 0.12
        clusters, current = [], [sorted_rows[0]]
        for row in sorted_rows[1:]:
            if row['y_center'] - current[-1]['y_center'] <= spacing:
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
        clusters.append(current)

        # Step 4 – Keep only meaningful clusters, compute fractional bounds
        buffer = image_height * 0.02
        regions = []
        for cluster in clusters:
            if len(cluster) >= self.min_rows or any(self._is_strong_ui_row(r) for r in cluster):
                min_y = min(r['y_start'] for r in cluster)
                max_y = max(r['y_end']   for r in cluster)
                regions.append((
                    max(0.0,  (min_y - buffer) / image_height),
                    min(1.0,  (max_y + buffer) / image_height),
                ))
        return regions
