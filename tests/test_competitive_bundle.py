import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from possession.competitive_bundle import input_signature


class BundleTests(unittest.TestCase):
    def test_input_fingerprints_do_not_depend_on_checkout_directory(self):
        results=[]
        for _ in range(2):
            with tempfile.TemporaryDirectory() as folder:
                def paths(source):
                    root=Path(folder)/source;root.mkdir(exist_ok=True)
                    values={}
                    for key in ['atd_raw','atd_meta','homography']:
                        p=root/f'{source}_{key}.txt';p.write_bytes((source+key).encode());values[key]=p
                    return values
                with patch('possession.competitive_bundle.asset_paths',side_effect=paths):results.append(input_signature())
        self.assertEqual(results[0],results[1]);self.assertEqual(len(results[0]),6)
        self.assertTrue(all(not Path(p).is_absolute() for p in results[0]))


if __name__=='__main__':unittest.main()
