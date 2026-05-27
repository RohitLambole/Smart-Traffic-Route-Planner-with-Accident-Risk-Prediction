"""
utils.py — Geohash Decoder & Helper Functions
Flipkart Gridlock Hackathon 2.0

Provides:
  - decode_geohash(geohash_str) → (latitude, longitude)
  - parse_timestamp(ts_str) → (hour, minute)
  - cyclic_encode(value, max_value) → (sin_component, cos_component)
"""

import math

# ============================================================
# GEOHASH DECODING (Pure Python — no external library needed)
# ============================================================

_BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(geohash_str):
    """
    Decode a geohash string into (latitude, longitude).
    
    Geohash is a hierarchical spatial encoding that recursively subdivides
    the world into a grid. Each character narrows the bounding box.
    
    Args:
        geohash_str: A geohash string (e.g., 'qp02z1')
    
    Returns:
        tuple: (latitude, longitude) as floats
    """
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    is_lon = True  # longitude bit comes first

    for char in geohash_str:
        val = _BASE32_MAP[char]
        for bit in range(4, -1, -1):  # 5 bits per character
            mid = (lon_range[0] + lon_range[1]) / 2 if is_lon else (lat_range[0] + lat_range[1]) / 2
            if is_lon:
                if val & (1 << bit):
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                if val & (1 << bit):
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon

    latitude = (lat_range[0] + lat_range[1]) / 2
    longitude = (lon_range[0] + lon_range[1]) / 2
    return latitude, longitude


def decode_geohash_batch(geohash_series):
    """
    Decode a pandas Series of geohash strings into two Series (lat, lon).
    Uses caching to avoid re-decoding duplicate geohashes.
    
    Args:
        geohash_series: pandas Series of geohash strings
    
    Returns:
        tuple: (latitudes_list, longitudes_list)
    """
    cache = {}
    lats = []
    lons = []
    for gh in geohash_series:
        if gh not in cache:
            cache[gh] = decode_geohash(gh)
        lat, lon = cache[gh]
        lats.append(lat)
        lons.append(lon)
    return lats, lons


# ============================================================
# TIMESTAMP PARSING
# ============================================================

def parse_timestamp(ts_str):
    """
    Parse a timestamp string like '8:30' or '0:0' into (hour, minute).
    
    Args:
        ts_str: String in format 'H:M' (e.g., '8:30', '0:0', '23:45')
    
    Returns:
        tuple: (hour, minute) as integers
    """
    parts = ts_str.split(':')
    return int(parts[0]), int(parts[1])


def time_slot_index(hour, minute):
    """
    Convert hour and minute to a unique time slot index (0–95).
    Each day has 96 fifteen-minute slots.
    
    Args:
        hour: Integer 0–23
        minute: Integer 0, 15, 30, or 45
    
    Returns:
        int: Slot index 0–95
    """
    return hour * 4 + minute // 15


# ============================================================
# CYCLIC ENCODING
# ============================================================

def cyclic_encode(value, max_value):
    """
    Encode a periodic/cyclic value using sin and cos.
    
    This ensures that values at the boundary (e.g., hour 23 and hour 0)
    are numerically close, unlike raw integer encoding.
    
    Args:
        value: The current value (e.g., hour=23)
        max_value: The period (e.g., 24 for hours)
    
    Returns:
        tuple: (sin_component, cos_component)
    """
    angle = 2 * math.pi * value / max_value
    return math.sin(angle), math.cos(angle)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == '__main__':
    # Test geohash decoding - qp02z1 should be somewhere in Bengaluru (~12.9N, 77.5E)
    lat, lon = decode_geohash('qp02z1')
    print(f"Geohash 'qp02z1' -> lat={lat:.4f}, lon={lon:.4f}")
    
    lat, lon = decode_geohash('qp094q')
    print(f"Geohash 'qp094q' -> lat={lat:.4f}, lon={lon:.4f}")
    
    # Test timestamp parsing
    h, m = parse_timestamp('8:30')
    print(f"\nTimestamp '8:30' -> hour={h}, minute={m}")
    print(f"Time slot index: {time_slot_index(h, m)}")
    
    # Test cyclic encoding
    sin_h, cos_h = cyclic_encode(23, 24)
    print(f"\nCyclic encode hour=23: sin={sin_h:.4f}, cos={cos_h:.4f}")
    sin_h, cos_h = cyclic_encode(0, 24)
    print(f"Cyclic encode hour=0:  sin={sin_h:.4f}, cos={cos_h:.4f}")
    print("(Notice: hour 23 and hour 0 are close in sin/cos space)")
