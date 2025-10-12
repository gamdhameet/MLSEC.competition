# Minimal multi-file extension of the didactic demo
# Usage:
#   python demo_multi.py <GOODWARE_DIR> <MALWARE_DIR> <UNKNOWN_1> [UNKNOWN_2 ...]
#
# Example (Windows):
#   python demo_multi.py gw_dir mw_dir C:\Windows\System32\taskmgr.exe

import os, sys, glob
import pefile
from sklearn import svm

# Helper function to open a PE file and count DLLs
def num_imported_dlls(path: str) -> int:
    """Return number of imported DLLs for a PE file. 0 if none or not parseable."""
    try:
        pe = pefile.PE(path, fast_load=True)
        # Ensure import directory is parsed even in fast_load mode
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']]
        )
        return len(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or [])
    except (pefile.PEFormatError, FileNotFoundError, PermissionError):
        # Not a valid PE or not accessible -> treat as 0 to keep demo simple
        return 0
    except Exception:
        return 0

def files_in_dir(d):
    # Grab all files (exe/dll/sys preferred, but any file will be tried)
    patterns = ["*.exe", "*.dll", "*.sys", "*"]
    seen = set()
    results = []
    for p in patterns:
        for f in glob.glob(os.path.join(d, p)):
            if os.path.isfile(f) and f.lower() not in seen:
                results.append(f)
                seen.add(f.lower())
    return results

if len(sys.argv) < 4:
    print("Usage: python demo_multi.py <GOODWARE_DIR> <MALWARE_DIR> <UNKNOWN_1> [UNKNOWN_2 ...]")
    sys.exit(1)

gw_dir, mw_dir = sys.argv[1], sys.argv[2]
unknown_files = sys.argv[3:]

gw_files = files_in_dir(gw_dir)
mw_files = files_in_dir(mw_dir)

if not gw_files or not mw_files:
    print("ERROR: no files found. Make sure GOODWARE_DIR and MALWARE_DIR contain PE files.")
    sys.exit(2)

# ---------- Build dataset ----------
X, Y = [], []

for f in gw_files:
    X.append([num_imported_dlls(f)])  # 1-D feature vector; keep it didactic
    Y.append(0)                       # 0 = goodware

for f in mw_files:
    X.append([num_imported_dlls(f)])
    Y.append(1)                       # 1 = malware

print(f"Training on {len(gw_files)} goodware + {len(mw_files)} malware files "
      f"(total {len(X)} samples).")

# ---------- Train ----------
clf = svm.SVC()   # same classifier as the demo; simple on purpose
clf.fit(X, Y)

# ---------- Predict ----------
label_map = ["goodware", "malware"]
for u in unknown_files:
    fcount = num_imported_dlls(u)
    pred = int(clf.predict([[fcount]])[0])
    print(f"{u} -> {label_map[pred]} (imports={fcount})")
