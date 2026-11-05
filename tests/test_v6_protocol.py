import json
from pathlib import Path
import unittest


class ProtocolTests(unittest.TestCase):
    def test_confirmation_is_disjoint_and_nonempty(self):
        p=json.loads((Path(__file__).resolve().parents[1]/'docs/PROTOCOLO_V6.json').read_text())
        train=set(p['native_train']);dev=set(p['native_development']);test=set(p['native_confirmation_locked'])
        self.assertTrue(test)
        self.assertTrue(train.isdisjoint(dev))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(dev.isdisjoint(test))
        self.assertEqual(len(train|dev|test),len(p['native_train'])+len(p['native_development'])+len(p['native_confirmation_locked']))


if __name__=='__main__':unittest.main()
