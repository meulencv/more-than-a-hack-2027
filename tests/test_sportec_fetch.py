import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.fetch_sportec_native import fetch, verified


class SportecFetchTests(unittest.TestCase):
    def test_exact_size_and_checksum_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'file.xml'
            path.write_bytes(b'test')
            record = dict(size=4, sha256=hashlib.sha256(b'test').hexdigest())
            self.assertTrue(verified(path, record))
            self.assertFalse(verified(path, dict(record, size=3)))
            self.assertFalse(verified(path, dict(record, sha256='0' * 64)))

    def test_path_traversal_rejected_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fetch(dict(name='../escape.xml', kind='tracking'), directory)


if __name__ == '__main__':
    unittest.main()
