import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from possession import v13_cv
from possession.v10_relational import sequences


class CrossMatchTests(unittest.TestCase):
    def test_closed_match_cannot_be_loaded(self):
        with self.assertRaisesRegex(ValueError,'Closed'):v13_cv.record('J03WR9','clean')

    def test_normalizer_uses_only_requested_training_games(self):
        calls=[]
        def record(role,game,mode):
            calls.append(game);v=5. if game=='J03WMX' else 999.
            return dict(slices=[(0,40)],nodes=np.zeros((40,2,14),dtype='float16'),ball_tokens=np.zeros((40,7),dtype='float32'),
                        team=np.array([0,1]),labels=np.zeros(40,dtype='int8'),direct=np.full((40,2),v,dtype='float32'),mirror=np.full((40,2),v,dtype='float32'))
        with patch('possession.v10_relational.data.record',side_effect=record):samples,norm=sequences(['J03WMX'])
        self.assertEqual(set(calls),{'J03WMX'});np.testing.assert_array_equal(norm.mean,[5.,5.]);self.assertEqual(len(samples),3)

    def test_partial_fold_cannot_be_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v13_cv,'ROOT',Path(folder)):
            root=Path(folder)/'fold0';root.mkdir();path=root/'results.json'
            path.write_text('[]');self.assertFalse(v13_cv.fold_complete(0))
            rows=[dict(game=g,mode=m,method=n) for g in v13_cv.settings()['test_folds'][0] for m in ('clean','moderate','stress') for n in ('baseline','combined')]
            path.write_text(json.dumps(rows));self.assertTrue(v13_cv.fold_complete(0))
            path.write_text(json.dumps(rows[:-1]));self.assertFalse(v13_cv.fold_complete(0))


if __name__=='__main__':unittest.main()
