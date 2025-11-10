#Model's takes these steps: 
#1.Take the raw PE bytes then checks the outside where it looks at things such as how random the data looks, how well the bytes compress, how much it looks like a base64 character string, etc. 
#2.If enough red flags are given,it then does in inside check where it tries to pull any payloads that is inside the PE file
#3.summarizes the outside and inside into numerical fingerprints
#4.scores them with a heuristic ratio that outputs a probability
#5.Makes a verdict based on that score

import os, zlib, base64, time
from typing import List, Tuple, Optional
import numpy as np

# ======== ADDED: optional dependencies and helpers (safe to ignore if not available) ========
import re
import pickle
import lzma

try:
    import pefile  # optional: for your own local experiments if you want to extend features later
except Exception:
    pefile = None

# Parameters to meet the challenge requirements
MAX_BYTES       = int(os.getenv("MAX_BYTES", str(2**21)))   # 2 MiB
TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "4.5"))
MZ_SLICE_LEN    = int(os.getenv("MZ_SLICE_LEN", "4096"))
MAX_CANDIDATES  = int(os.getenv("MAX_CANDIDATES", "4"))
# decision threshold (for the simple heuristic or your optional sklearn)
DECISION = float(os.getenv("DECISION", "0.60"))
# Optional: point to a sklearn pickle with {"scaler":..., "classifier":..., "threshold":...}
MODEL_PATH = os.getenv("MODEL_PATH", "")  # leave empty to use heuristic

_loaded = False
_scaler = None
_clf = None
_threshold = DECISION  

# ======== ADDED: load sklearn model if provided (keeps your params; uses them only if MODEL_PATH set) ========
def _load_model(path: str):
    global _loaded, _scaler, _clf, _threshold
    if not path:
        return
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            _scaler = obj.get("scaler", None)
            _clf = obj.get("classifier", obj.get("clf", None))
            _threshold = float(obj.get("threshold", DECISION))
        else:
            # allow pickle of a bare estimator
            _clf = obj
        _loaded = _clf is not None
    except Exception:
        _loaded = False

if MODEL_PATH:
    _load_model(MODEL_PATH)

#These functions are used both for outside and inside processes and measure basic bayte-level properties
def entropy(b: bytes) -> float: #Sees how random the data looks
    if not b: return 0.0
    arr = np.frombuffer(b, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts / (counts.sum() + 1e-9)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())

def printable_ratio(b: bytes) -> float: #Calculates what fraction of bytes that are printable ASCII characters
    if not b: return 0.0
    arr = np.frombuffer(b, dtype=np.uint8)
    return float(((arr >= 32) & (arr <= 126)).sum() / arr.size)

def compress_ratio(b: bytes) -> float: #Sees how well the bytes compresse
    if not b: return 1.0
    try:
        c = zlib.compress(b, 6)
        return len(c) / len(b)
    except Exception:
        return 1.0

def count_mz(b: bytes, cap:int=6) -> int: #Sees how many times "MZ" appears which is the header of most Windows PE files
    cnt, i = 0, b.find(b"MZ")
    while i != -1 and cnt < cap:
        cnt += 1
        i = b.find(b"MZ", i+1)
    return cnt

def looks_like_base64_chunk(b: bytes) -> bool: #Scans for Base64 Characters (That heuristic for encoded data from Seminar)
    if len(b) < 128: return False
    base = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    streak = 0
    for x in b:
        if x in base:
            streak += 1
            if streak >= 128:
                return True
        else:
            streak = 0
    return False

def suspicious(blob: bytes) -> bool: #This suspicious function just decides if there are enough red flags to check the inside using the functions above
    ent = entropy(blob)
    pr  = printable_ratio(blob)
    cr  = compress_ratio(blob)
    mzs = count_mz(blob)
    has_tags = (b"RESZ" in blob) or (b"PACK:" in blob) or (b"BASE64:" in blob) or looks_like_base64_chunk(blob)
    return (ent > 7.2) or (pr < 0.25) or (cr > 0.95) or (mzs >= 2) or has_tags

# ======== ADDED: safe decompress helpers & extra decoders to catch more dropper tricks ========
_MAX_DECOMP = 5 * 1024 * 1024  # safety cap for decompression bombs

def _safe_zlib(data: bytes) -> Optional[bytes]:
    try:
        out = zlib.decompress(data)
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        pass
    return None

def _safe_lzma(data: bytes) -> Optional[bytes]:
    try:
        out = lzma.decompress(data)
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        pass
    return None

def _safe_gzip(data: bytes) -> Optional[bytes]:
    try:
        import gzip
        out = gzip.decompress(data)
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        pass
    return None

def _try_b64_long(blob: bytes) -> Optional[bytes]:
    m = re.search(rb"([A-Za-z0-9+/=\r\n]{120,})", blob)
    if not m: return None
    s = m.group(1)
    try:
        out = base64.b64decode(s, validate=True)
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        return None
    return None

def _try_b64_url(blob: bytes) -> Optional[bytes]:
    m = re.search(rb"([A-Za-z0-9_\-]{120,})", blob)
    if not m: return None
    s = m.group(1)
    try:
        out = base64.urlsafe_b64decode(s + b'=' * (-len(s) % 4))
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        return None
    return None

def _try_hex(blob: bytes) -> Optional[bytes]:
    m = re.search(rb"([0-9a-fA-F]{128,})", blob)
    if not m: return None
    s = m.group(1)
    try:
        out = bytes.fromhex(s.decode("ascii"))
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        return None
    return None

def _try_gzip_lzma(blob: bytes) -> Optional[bytes]:
    return _safe_gzip(blob) or _safe_lzma(blob)

# Bounded extraction (inside peek) 
#These are functions that will be used if suspicious if activated and tries to pull hidden payloads that may be in a dropper through trial and error
def find_mz_offsets(blob: bytes, max_hits:int=6) -> List[int]: #Finds locations of “MZ” headers inside the file.
    offs, i = [], blob.find(b"MZ")
    while i != -1 and len(offs) < max_hits:
        offs.append(i)
        i = blob.find(b"MZ", i+1)
    return offs

def try_resz(blob: bytes) -> Optional[bytes]: #Looks for a fake RESZ header that might contain a compressed block.
    idx = blob.find(b"RESZ")
    if idx == -1 or idx + 8 > len(blob): return None
    size = int.from_bytes(blob[idx+4:idx+8], "little")
    st = idx + 8; ed = st + size
    if ed > len(blob): return None
    try:
        return zlib.decompress(blob[st:ed])
    except Exception:
        return None

def try_b64(blob: bytes) -> Optional[bytes]: #We see if "BASE64:" is present then decode it if so
    idx = blob.find(b"BASE64:")
    if idx == -1: return None
    st = idx + len(b"BASE64:")
    ed = blob.find(b"\n", st)
    if ed == -1: ed = len(blob)
    try:
        return base64.b64decode(blob[st:ed], validate=True)
    except Exception:
        return None

def try_xor(blob: bytes) -> Optional[bytes]: #We see if "XORv1<key>" is found then decode it using that key
    idx = blob.find(b"XORv1")
    if idx == -1 or idx + 6 > len(blob): return None
    key = blob[idx+5]
    st = idx + 6
    ed = blob.find(b"\n", st)
    if ed == -1: ed = len(blob)
    seg = blob[st:ed]
    return bytes([x ^ key for x in seg])

def try_pack(blob: bytes) -> Optional[bytes]: #See if we have "PACK:" then decode it if so
    idx = blob.find(b"PACK:")
    if idx == -1: return None
    st = idx + len(b"PACK:")
    ed = blob.find(b"\n", st)
    if ed == -1: ed = len(blob)
    try:
        inner = base64.b64decode(blob[st:ed], validate=True)
        return zlib.decompress(inner)
    except Exception:
        return None

# ======== ADDED: extra candidate helpers (kept separate so your originals remain untouched) ========
def _try_pack_safe(blob: bytes) -> Optional[bytes]:
    idx = blob.find(b"PACK:")
    if idx == -1: return None
    st = idx + len(b"PACK:")
    ed = blob.find(b"\n", st)
    if ed == -1: ed = len(blob)
    try:
        inner = base64.b64decode(blob[st:ed], validate=True)
        return _safe_zlib(inner) or _safe_lzma(inner) or inner
    except Exception:
        return None

def _try_b64_line_safe(blob: bytes) -> Optional[bytes]:
    idx = blob.find(b"BASE64:")
    if idx == -1: return None
    st = idx + len(b"BASE64:")
    ed = blob.find(b"\n", st)
    if ed == -1: ed = len(blob)
    try:
        out = base64.b64decode(blob[st:ed], validate=True)
        if len(out) <= _MAX_DECOMP:
            return out
    except Exception:
        return None
    return None

def _try_xor_guess(blob: bytes) -> Optional[bytes]:
    # light repeating-XOR guesser (bounded to MZ_SLICE_LEN for speed)
    sample = blob[:min(len(blob), MZ_SLICE_LEN)]
    best = None
    best_pr = 0.0
    for klen in range(1, 9):
        # heuristic: assume plaintext had many spaces -> guess key from first klen bytes
        key = bytes([sample[i] ^ 0x20 for i in range(min(klen, len(sample)))])
        if not key: continue
        cand = bytes([b ^ key[i % len(key)] for i, b in enumerate(sample)])
        pr = printable_ratio(cand)
        if pr > 0.30 and pr > best_pr:
            best = cand
            best_pr = pr
    return best

def extract_candidates(blob: bytes, max_cands:int=MAX_CANDIDATES) -> List[Tuple[bytes, str]]:
    cands: List[Tuple[bytes, str]] = []
    # 1) Short slices at MZ (bounded) — typical for embedded executables
    for off in find_mz_offsets(blob)[:3]:
        end = min(len(blob), off + MZ_SLICE_LEN)
        cands.append((blob[off:end], f"mz_{off}"))
        if len(cands) >= max_cands: return cands
    # 2) Tagged/encoded payloads which are common dropper tricks after putting MZ since MZ is common for goodware
    for fn, name in ((try_resz,"resz"), (try_b64,"b64"), (try_xor,"xor"), (try_pack,"pack")):
        out = fn(blob)
        if out:
            cands.append((out, name))
            if len(cands) >= max_cands: return cands
    # ======== ADDED: more decoders below while keeping your originals above ========
    for fn, name in ((_try_b64_line_safe,"b64_line_safe"),
                     (_try_b64_long,"b64_long"),
                     (_try_b64_url,"b64_url"),
                     (_try_hex,"hex"),
                     (_try_pack_safe,"pack_safe"),
                     (_try_gzip_lzma,"gzip_lzma"),
                     (_try_xor_guess,"xor_guess")):
        try:
            out = fn(blob)
        except Exception:
            out = None
        if out:
            cands.append((out, name))
            if len(cands) >= max_cands: return cands
    # Fallback: raw blob
    if not cands:
        cands.append((blob, "raw"))
    return cands

# Feature extraction
def byte_histogram(b: bytes, bins:int=16) -> np.ndarray:
    if not b: return np.zeros(bins, dtype=np.float32)
    arr = np.frombuffer(b, dtype=np.uint8)
    hist, _ = np.histogram(arr, bins=bins, range=(0,256))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist

def suspicious_string_count(b: bytes) -> float: #counts how many suspicious substrings exist, likely can be expanded as we can put in more
    c = 0.0
    for t in (b"cmd.exe", b"powershell", b"rundll32", b"regsvr32", b"certutil", b"bitsadmin"):
        if t in b: c += 1.0
    # ======== ADDED: a few more without changing your interface (still a float count) ========
    for t in (b"VirtualAlloc", b"VirtualProtect", b"CreateRemoteThread", b"LoadLibrary", b"PowerShell -"):
        if t in b: c += 1.0
    return c

def features_for_blob(b: bytes) -> np.ndarray: #Gets the features for one blob
    size = float(len(b))
    ent  = entropy(b)
    pr   = printable_ratio(b)
    cr   = compress_ratio(b)
    mzs  = float(count_mz(b))
    sus  = suspicious_string_count(b)
    hist = byte_histogram(b, bins=16)
    return np.concatenate([np.array([size, ent, pr, cr, mzs, sus], dtype=np.float32), hist], axis=0)

def make_feature_vector(outside_blob: bytes, inside_best: Optional[bytes]) -> np.ndarray:
    f_out = features_for_blob(outside_blob)  # 22
    f_in  = features_for_blob(inside_best) if inside_best else np.zeros_like(f_out)
    return np.concatenate([f_out, f_in], axis=0).reshape(1, -1)  # (1,44)

def score_with_model(X: np.ndarray) -> float:
    """Return P(malicious) in [0,1]. Uses a simple heuristic.""" 
    #The scores can be changed depending on how important some of these parameters are
    if not _loaded:
        # ======== CHANGED: still a simple heuristic, but now also considers the "inside" half if present
        # outside
        ent, pr, cr, mzs, sus = float(X[0,1]), float(X[0,2]), float(X[0,3]), float(X[0,4]), float(X[0,5])
        score = 0.0
        if ent > 7.2: score += 0.5
        if cr  > 0.95: score += 0.2
        if pr  < 0.25: score += 0.2
        if mzs >= 2:   score += 0.1
        if sus > 0:    score += 0.1
        # inside (indices 22..43 mirror 0..21)
        i_ent, i_pr, i_cr, i_mzs, i_sus = float(X[0,23]), float(X[0,24]), float(X[0,25]), float(X[0,26]), float(X[0,27])
        # add a smaller weight for the inside signals
        if i_ent > 7.2: score += 0.15
        if i_cr  > 0.95: score += 0.05
        if i_pr  < 0.25: score += 0.05
        if i_mzs >= 1:   score += 0.05
        if i_sus > 0:    score += 0.05
        return max(0.0, min(1.0, score))
    # sklearn path
    Xs = _scaler.transform(X) if _scaler is not None else X
    if hasattr(_clf, "predict_proba"):
        return float(_clf.predict_proba(Xs)[0, 1])
    return float(_clf.predict(Xs)[0])  # 0/1 fallback

# This is where the model actually does its thing now that we have all of the functions it will use
class DropperAwareModel:
    def __init__(self):
        # you could read env here if needed
        pass

    def predict(self, bytez: bytes) -> int:
        # Size/empty guards (contest rules: >2 MiB effectively benign)
        if not bytez or len(bytez) == 0 or len(bytez) > MAX_BYTES:
            return 0

        t0 = time.monotonic()

        # Outside always considered
        inside_best = None

        # Only peek inside if it looks like a dropper and we have time
        if suspicious(bytez):
            best_ent = -1.0
            # ======== CHANGED: use improved extractor above (which still includes all your original decoders)
            for cand, _name in extract_candidates(bytez, max_cands=MAX_CANDIDATES):
                if time.monotonic() - t0 > TIME_BUDGET_SEC:
                    break
                e = entropy(cand)
                # ======== CHANGED: tie-break with suspicious strings to prefer "spicier" candidates
                tie = suspicious_string_count(cand) * 0.05 - printable_ratio(cand) * 0.02
                if (e + tie) > best_ent:
                    best_ent = e + tie
                    inside_best = cand

        # Build features (outside + best-inside)
        X = make_feature_vector(bytez, inside_best)

        # Score -> probability of malicious
        p_mal = score_with_model(X)

        # Decide using threshold (from env or sklearn pickle)
        thr = _threshold
        return 1 if p_mal >= thr else 0


# Helper: extraction wrapper (kept near class for readability)
def extract_candidates(blob: bytes, max_cands:int=MAX_CANDIDATES) -> List[Tuple[bytes, str]]:
    return _extract_candidates_impl(blob, max_cands)

def _extract_candidates_impl(blob: bytes, max_cands:int) -> List[Tuple[bytes, str]]:
    cands: List[Tuple[bytes, str]] = []
    for off in find_mz_offsets(blob)[:3]:
        end = min(len(blob), off + MZ_SLICE_LEN)
        cands.append((blob[off:end], f"mz_{off}"))
        if len(cands) >= max_cands: return cands
    for fn, name in ((try_resz,"resz"), (try_b64,"b64"), (try_xor,"xor"), (try_pack,"pack"),
                     (_try_b64_line_safe,"b64_line_safe"),
                     (_try_b64_long,"b64_long"),
                     (_try_b64_url,"b64_url"),
                     (_try_hex,"hex"),
                     (_try_pack_safe,"pack_safe"),
                     (_try_gzip_lzma,"gzip_lzma"),
                     (_try_xor_guess,"xor_guess")):
        out = fn(blob)
        if out:
            cands.append((out, name))
            if len(cands) >= max_cands: return cands
    if not cands:
        cands.append((blob, "raw"))
    return cands
