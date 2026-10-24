"""Download the CirCor DigiScope dataset (v1.0.3) from PhysioNet's open S3 mirror.

The official zip at physionet.org is often very slow; the S3 mirror
(s3://physionet-open, public, no credentials) has the same files.

Usage:
    python download_data.py            # -> data/circor/
    python download_data.py --out DIR --workers 32

Re-running is safe: files already present with the right size are skipped.
"""
import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "circor-heart-sound/1.0.3/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys():
    """Yield (key, size) for every object under PREFIX."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX}
        if token:
            params["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(params)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for c in root.findall("s3:Contents", NS):
            yield c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)
        if root.find("s3:IsTruncated", NS).text != "true":
            break
        token = root.find("s3:NextContinuationToken", NS).text


def fetch(key, size, out_dir):
    dest = out_dir / key[len(PREFIX):]
    if dest.exists() and dest.stat().st_size == size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.rename(dest)
            return True
        except Exception:
            if attempt == 4:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path(__file__).parent / "data" / "circor", type=Path)
    ap.add_argument("--workers", default=32, type=int)
    args = ap.parse_args()

    keys = [(k, s) for k, s in list_keys() if not k.endswith("/")]
    total = sum(s for _, s in keys)
    print(f"{len(keys)} files, {total / 1e6:.0f} MB -> {args.out}")

    done = new = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch, k, s, args.out) for k, s in keys]
        for fut in as_completed(futures):
            new += fut.result()
            done += 1
            if done % 1000 == 0 or done == len(keys):
                print(f"  {done}/{len(keys)} ({new} downloaded)", flush=True)


if __name__ == "__main__":
    main()