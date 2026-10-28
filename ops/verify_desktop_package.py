"""Synchronize this project's handoff copy and verify every source file byte.

Run with --destination after the first APFS clone. Never removes source or
destination-only files and never rewrites historical provenance or code.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def inventory(root):
    files={};links={}
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in dirs+names:
            f=Path(folder)/name
            if f.is_symlink():links[f.relative_to(root).as_posix()]=os.readlink(f)
        for name in names:
            f=Path(folder)/name
            if f.is_symlink():continue
            rel=f.relative_to(root).as_posix()
            if rel in ['DESKTOP_PACKAGE_MANIFEST.json','DESKTOP_PACKAGE_VERIFICATION.json']:continue
            s=f.stat();files[rel]=(s.st_size,s.st_mtime_ns)
    return files,links


class Links(HTMLParser):
    def __init__(self):super().__init__();self.values=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value:self.values.append(value)
            if tag=='img' and key=='src':self.images.append(value)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True);args=ap.parse_args()
    src=Path(__file__).resolve().parents[1];dst=Path(args.destination).resolve()
    if src==dst or not dst.is_dir():raise SystemExit('Existing independent destination required')
    start=time.monotonic();before,links=inventory(src);copied=[];records=[]
    for rel,(size,mtime) in sorted(before.items()):
        a,b=src/rel,dst/rel
        if not b.exists() or (b.stat().st_size,b.stat().st_mtime_ns)!=(size,mtime):
            b.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['/bin/cp','-cp',str(a),str(b)],check=True);copied.append(rel)
    for rel,target in links.items():
        b=dst/rel
        if not b.is_symlink() or os.readlink(b)!=target:
            raise RuntimeError('Symlink copy differs: '+rel)
        if not b.exists() or not b.resolve().is_relative_to(dst):
            raise RuntimeError('Missing or external dependency: '+rel)
    print(json.dumps({'phase':'synced','files':len(before),'updated_files':len(copied)}),flush=True)
    done_bytes=0;last_progress=0
    for rel,(size,mtime) in sorted(before.items()):
        a,b=src/rel,dst/rel;ha=digest(a);hb=digest(b)
        if ha!=hb:raise RuntimeError('Content mismatch: '+rel)
        s=a.stat()
        if (s.st_size,s.st_mtime_ns)!=(size,mtime):raise RuntimeError('Source changed during hash: '+rel)
        records.append({'path':rel,'bytes':size,'sha256':ha})
        done_bytes+=size
        if done_bytes-last_progress>3_000_000_000:
            print(json.dumps({'phase':'hashing','verified_bytes':done_bytes,'verified_files':len(records)}),flush=True);last_progress=done_bytes
    after,after_links=inventory(src)
    if before!=after or links!=after_links:raise RuntimeError('Source changed during verification; rerun for a coherent package')
    parser=Links();parser.feed((dst/'START_HERE.html').read_text())
    missing=[]
    for value in parser.values:
        u=urlparse(value)
        if u.scheme or not u.path:continue
        if not (dst/unquote(u.path)).exists():missing.append(value)
    if missing:raise RuntimeError('Broken review-page references: '+repr(missing))
    created=datetime.now(timezone.utc).isoformat()
    manifest={'created_at_utc':created,'source':str(src),'destination':str(dst),
              'method':'Independent APFS clone and explicit byte-hash verification; original preserved',
              'source_regular_files':len(records),'source_logical_bytes':done_bytes,
              'symlinks':links,'files':records,
              'scope':'All regular files and both internal relative links in the project; no external symlink dependencies'}
    path=dst/'DESKTOP_PACKAGE_MANIFEST.json';path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    receipt={'status':'passed','created_at_utc':created,'destination':str(dst),
             'verified_files':len(records),'verified_bytes':done_bytes,'sha256_mismatches':0,
             'external_symlink_dependencies':0,'internal_symlinks_verified':len(links),
             'portal_image_count':len(parser.images),'portal_missing_local_links':missing,
             'source_unchanged_during_verification':True,'manifest_sha256':digest(path),
             'elapsed_seconds':time.monotonic()-start,
             'preserved_historical_absolute_provenance':True,
             'browser_visual_preview':'Not performed: in-app browser disallows file URLs; relative links checked locally',
             'automated_training_or_submission':False}
    (dst/'DESKTOP_PACKAGE_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
