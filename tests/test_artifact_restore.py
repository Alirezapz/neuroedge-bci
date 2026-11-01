"""Software-only archive fixtures; no synthetic EEG or research measurements."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('artifact_restore', Path(__file__).parents[1] / 'scripts/restore_model_bundle.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RestoreTests(unittest.TestCase):
    def fixture(self, root, name='runs/test/record.txt'):
        data = b'Clearly labeled software fixture\n'
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(name, data)
        raw = memory.getvalue()
        parts = []
        (root / 'artifacts').mkdir()
        for index, block in enumerate([raw[:50], raw[50:]]):
            path = f'artifacts/fixture.part{index:03}'
            (root / path).write_bytes(block)
            parts.append({'path': path, 'size': len(block), 'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(block)).encode() + b'\0' + block).hexdigest()})
        manifest = {'schema_version': 1, 'archives': [{'name': 'fixture', 'parts': parts}],
                    'files': {name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}
        path = root / 'artifacts/MANIFEST.json'
        path.write_text(json.dumps(manifest))
        return path, data

    def test_verify_restore_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, expected = self.fixture(root)
            self.assertEqual(MODULE.restore(root, manifest, True)['verified_files'], 1)
            self.assertFalse((root / 'runs').exists())
            self.assertEqual(MODULE.restore(root, manifest)['restored_files'], 1)
            self.assertEqual((root / 'runs/test/record.txt').read_bytes(), expected)
            self.assertEqual(MODULE.restore(root, manifest)['unchanged_files'], 1)

    def test_corrupted_part_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            part = root / 'artifacts/fixture.part001'
            block = part.read_bytes()
            part.write_bytes(block[:-1] + bytes([block[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'part hash mismatch'):
                MODULE.restore(root, manifest)

    def test_changed_local_file_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            target = root / 'runs/test/record.txt'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'Local work')
            with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
                MODULE.restore(root, manifest)
            self.assertEqual(target.read_bytes(), b'Local work')

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root, '../outside.txt')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                MODULE.restore(root, manifest)

    def test_manifest_file_hash_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self.fixture(root)
            content = json.loads(manifest.read_text())
            content['files']['runs/test/record.txt']['sha256'] = '0' * 64
            manifest.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, 'file hash mismatch'):
                MODULE.restore(root, manifest)
            self.assertFalse((root / 'runs/test/record.txt').exists())


if __name__ == '__main__':
    unittest.main()
