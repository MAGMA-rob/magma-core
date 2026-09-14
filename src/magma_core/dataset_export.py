"""Shared JSON persistence, task partitions and verified export manifests."""
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Mapping

from .protocol.agent_export import EXPORT_VERSION


def write_json(path: Path, value: Any) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f'Cannot write dataset through a symlink: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.is_symlink():
        raise ValueError(f'Invalid temporary dataset path: {temporary}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def export_signature(sources: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    return sha256(json.dumps([EXPORT_VERSION, sources, settings], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def completed_export(folder: Path, signature: str) -> bool:
    path = folder / 'export_manifest.json'
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text())
        if not manifest.get('complete') or manifest.get('signature') != signature:
            return False
        for name, digest in manifest['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                return False
            target = folder / relative
            if target.is_symlink() or not target.is_file() or sha256(target.read_bytes()).hexdigest() != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def write_datasets(folder: Path, tasks: Mapping[str, Mapping[str, list[dict[str, Any]]]], *,
                   filenames: Mapping[str, str], signature: str, settings: Mapping[str, Any],
                   failures: list[dict[str, Any]], validation_ratio: float = .05,
                   split_seed: int = 42) -> dict[str, Path]:
    if not 0 < validation_ratio < 1:
        raise ValueError('validation_ratio must be strictly between 0 and 1')
    task_ids = sorted(tasks)
    random.Random(split_seed).shuffle(task_ids)
    count = min(len(task_ids) - 1, max(1, round(len(task_ids) * validation_ratio))) if len(task_ids) > 1 else 0
    validation = set(task_ids[:count])
    files: dict[str, Path] = {}
    rows_count: dict[str, int] = {}
    for channel, filename in filenames.items():
        if Path(filename).name != filename or not filename.endswith('.json'):
            raise ValueError(f'Invalid dataset filename: {filename!r}')
        rows = [row for task in sorted(tasks) for row in tasks[task].get(channel, [])]
        target = folder / filename
        write_json(target, rows)
        files[filename] = target
        rows_count[channel] = len(rows)
        for split in ('train', 'validation'):
            selected = [row for task in sorted(tasks) if (task in validation) == (split == 'validation')
                        for row in tasks[task].get(channel, [])]
            name = f'{Path(filename).stem}_{split}.json'
            write_json(folder / name, selected)
            files[name] = folder / name
    write_json(folder / 'export_failures.json', failures)
    files['export_failures.json'] = folder / 'export_failures.json'
    write_json(folder / 'export_manifest.json', {
        'format_version': EXPORT_VERSION, 'complete': not failures, 'signature': signature,
        'settings': dict(settings), 'rows': rows_count,
        'split': {'train': sorted(set(task_ids) - validation), 'validation': sorted(validation)},
        'files': {name: sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
    })
    return files
