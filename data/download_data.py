"""
Download the Majdak, Goupell & Laback (2010) localisation-training data set
from the Auditory Modeling Toolbox (AMT) auxiliary data and verify its checksum.

Usage (from the repository root):
    python data/download_data.py

The file is saved as data/data.mat (about 1.2 MB). The data are distributed by
AMT (https://amtoolbox.org) on behalf of the original authors; this repository
does not redistribute them.

Reference:
    Majdak, P., Goupell, M. J., & Laback, B. (2010). 3-D localization of virtual
    sound sources: Effects of visual environment, pointing method, and training.
    Attention, Perception, & Psychophysics, 72(2), 454-469.
    https://doi.org/10.3758/APP.72.2.454
"""
import hashlib
import os
import sys
import urllib.request

URL = "https://amtoolbox.org/amt-1.0.0/auxdata/majdak2010/data.mat"
SHA256 = "d3df2bd84b565b7e55833dd40d21c7090b70714473bf54e92186c8b38082cee6"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.mat")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if os.path.exists(DEST) and sha256(DEST) == SHA256:
        print(f"data.mat already present and verified: {DEST}")
        return 0
    print(f"Downloading {URL} ...")
    tmp = DEST + ".part"
    urllib.request.urlretrieve(URL, tmp)
    digest = sha256(tmp)
    if digest != SHA256:
        os.remove(tmp)
        print("Checksum mismatch:", digest, "expected", SHA256, file=sys.stderr)
        return 1
    os.replace(tmp, DEST)
    print(f"Saved and verified: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
