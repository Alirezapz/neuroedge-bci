#!/usr/bin/env python3
"""Verify and restore the exact NeuroEdge research archives (standard library)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes the repository: {name}')
    return target

def digest_file(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def restore(root: Path, manifest_path: Path, verify_only: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported archive manifest version')
    expected = manifest['files']
    seen = set()
    restored = unchanged = 0
    for archive in manifest['archives']:
        with tempfile.TemporaryFile() as joined:
            for part in archive['parts']:
                path = safe_path(root, part['path'])
                size = path.stat().st_size
                if size != part['size']:
                    raise ValueError(f'Archive part size mismatch: {path}')
                git_hash = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
                with path.open('rb') as handle:
                    while block := handle.read(1024 * 1024):
                        git_hash.update(block)
                        joined.write(block)
                if git_hash.hexdigest() != part['git_blob_sha1']:
                    raise ValueError(f'Archive part hash mismatch: {path}')
            joined.seek(0)
            with zipfile.ZipFile(joined) as bundle:
                for info in bundle.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename.removeprefix('neuroedge/')
                    destination = safe_path(root, name)
                    if stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError(f'Symlink entries are not permitted: {name}')
                    if name in seen or name not in expected:
                        raise ValueError(f'Duplicate or unexpected archive entry: {name}')
                    meta = expected[name]
                    if info.file_size != meta['size']:
                        raise ValueError(f'Restored file size mismatch: {name}')
                    if destination.exists() and (destination.stat().st_size != meta['size']
                                                or digest_file(destination) != meta['sha256']):
                        raise ValueError(f'Refusing to replace a different local file: {name}')
                    temporary_name = None
                    try:
                        if not verify_only and not destination.exists():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                                temporary_name = output.name
                        checksum = hashlib.sha256()
                        with bundle.open(info) as handle:
                            while block := handle.read(1024 * 1024):
                                checksum.update(block)
                                if temporary_name:
                                    with open(temporary_name, 'ab') as output:
                                        output.write(block)
                        if checksum.hexdigest() != meta['sha256']:
                            raise ValueError(f'Restored file hash mismatch: {name}')
                        if temporary_name:
                            os.replace(temporary_name, destination)
                            temporary_name = None
                            restored += 1
                        elif not verify_only:
                            unchanged += 1
                    finally:
                        if temporary_name:
                            Path(temporary_name).unlink(missing_ok=True)
                    seen.add(name)
        print(f"Verified {archive['name']}: {len(archive['parts'])} archive parts")
    if seen != set(expected):
        raise ValueError(f'Missing archive entries: {sorted(set(expected) - seen)[:10]}')
    return {'verified_files': len(seen), 'restored_files': restored,
            'unchanged_files': unchanged, 'verify_only': verify_only}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Validate archives without extracting files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(restore(root, root / 'artifacts/MANIFEST.json', args.verify_only), indent=2))

if __name__ == '__main__':
    main()
