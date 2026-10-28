"""Three-way project handoff update that preserves IDE/user edits.

Uses the previous desktop manifest as baseline. Conflicts are reported and
never overwritten. Source data, archived artifacts, and historical paths stay
unchanged. Finder metadata is excluded from content parity.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,subprocess,time
from urllib.parse import urlparse,unquote
from ops.verify_desktop_package import digest,inventory,Links


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    src=Path(__file__).resolve().parents[1];dst=a.destination.resolve()
    if dst==src or not dst.is_dir():raise ValueError('Existing independent desktop package required')
    baseline=json.loads((dst/'DESKTOP_PACKAGE_MANIFEST.json').read_text())
    old={r['path']:r['sha256'] for r in baseline['files']}
    before,links=inventory(src);records=[];updated=[];preserved=[];conflicts=[];total=0;start=time.monotonic()
    for rel,(size,mtime) in sorted(before.items()):
        if Path(rel).name=='.DS_Store':continue
        source,target=src/rel,dst/rel;source_hash=digest(source)
        target_hash=digest(target) if target.is_file() and not target.is_symlink() else None
        previous=old.get(rel)
        if target_hash!=source_hash:
            if previous is not None and target_hash is not None and target_hash!=previous:
                if source_hash==previous:preserved.append(rel)
                else:conflicts.append(rel)
            elif previous is not None and target_hash is None:
                preserved.append(rel+' (deleted in desktop copy)')
            elif previous is None and target_hash is not None:
                conflicts.append(rel)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                subprocess.run(['/bin/cp','-cp',str(source),str(target)],check=True)
                target_hash=digest(target);updated.append(rel)
                if target_hash!=source_hash:raise RuntimeError('Copy hash mismatch: '+rel)
        record={'path':rel,'bytes':size,'sha256':source_hash}
        if target_hash!=source_hash:record.update(desktop_sha256=target_hash,user_change_preserved=True)
        records.append(record);total+=size
    after,after_links=inventory(src)
    if before!=after or links!=after_links:raise RuntimeError('Source changed during handoff verification; rerun after writes finish')
    for rel,target in links.items():
        local=dst/rel
        if not local.is_symlink() or os.readlink(local)!=target or not local.exists() or not local.resolve().is_relative_to(dst):
            raise RuntimeError('Internal link identity changed: '+rel)
    parsed=Links();parsed.feed((dst/'START_HERE.html').read_text())
    missing=[v for v in parsed.values if not urlparse(v).scheme and urlparse(v).path and not (dst/unquote(urlparse(v).path)).exists()]
    if missing:raise RuntimeError('Missing review-page links: '+repr(missing))
    created=datetime.now(timezone.utc).isoformat()
    manifest={'created_at_utc':created,'source':str(src),'destination':str(dst),
        'method':'Three-way update with prior manifest; user edits preserved',
        'source_regular_files':len(records),'source_logical_bytes':total,'symlinks':links,'files':records,
        'preserved_user_changes':preserved,'conflicts':conflicts,'excluded_metadata':['.DS_Store']}
    (dst/'DESKTOP_PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result={'status':'passed' if not conflicts and not preserved else 'user_changes_preserved_review_needed',
        'created_at_utc':created,'verified_files':len(records),'verified_bytes':total,
        'updated_files':updated,'preserved_user_changes':preserved,'conflicts':conflicts,
        'portal_image_count':len(parsed.images),'portal_missing_local_links':missing,
        'internal_symlinks_verified':len(links),'external_symlink_dependencies':0,
        'source_unchanged_during_verification':True,'manifest_sha256':digest(dst/'DESKTOP_PACKAGE_MANIFEST.json'),
        'elapsed_seconds':time.monotonic()-start,'automated_training_or_submission':False}
    (dst/'DESKTOP_PACKAGE_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
