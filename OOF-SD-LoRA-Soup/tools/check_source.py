"""Verify the source-release manifest, syntax and README links without model assets."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    count = 0
    manifest = ROOT / 'MANIFEST.sha256'
    if not manifest.is_file():
        raise SystemExit('MANIFEST.sha256 is missing')
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        digest, name = line.split('  ', 1)
        path = ROOT / name
        count += 1
        if not path.is_file():
            failures.append(f'missing: {name}')
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f'hash mismatch: {name}')
        try:
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError, ValueError) as exc:
            failures.append(f'{name}: {exc}')
    for name in ('README.md', 'README.ja.md', 'docs/ASSETS.md'):
        path = ROOT / name
        if not path.is_file():
            failures.append(f'missing document: {name}')
            continue
        content = path.read_text(encoding='utf-8')
        if content.count('```') % 2:
            failures.append(f'unclosed code block: {name}')
        for link in re.findall(r'\]\(([^)]+)\)', content):
            if link.startswith(('http://', 'https://', '#', 'mailto:')):
                continue
            target = unquote(link.split('#', 1)[0])
            if not (path.parent / target).exists():
                failures.append(f'broken file link in {name}: {link}')
    if failures:
        raise SystemExit('Source check failed:\n' + '\n'.join(failures))
    print(f'Source release verified: {count} files; Python/JSON syntax and document file links valid.')
    print('Model weights, datasets and experiment reproduction were not checked.')


if __name__ == '__main__':
    main()
