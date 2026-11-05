import unittest
from possession.sportec_native import native_label


class NativeReferenceTests(unittest.TestCase):
    def test_native_label_mapping(self):
        self.assertEqual(native_label('1', '1'), 0)
        self.assertEqual(native_label('1', '2'), 1)
        self.assertEqual(native_label('0', '1'), 2)
        self.assertEqual(native_label('0', '0'), 2)
        for status, possession in [('1', '0'), ('1', None), (None, '1'), ('2', '2')]:
            self.assertEqual(native_label(status, possession), -1)


if __name__ == '__main__':
    unittest.main()
