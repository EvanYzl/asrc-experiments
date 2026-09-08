"""Guard against reporting one run as a multi-seed estimate."""
import unittest

from publication_policy import required_seeds,summarize
from table_scope import cell_scope


class PublicationPolicyTests(unittest.TestCase):
    def setUp(self):
        methods=['LSMGA','DMKGC','IMKGC'];datasets=['dbp5l','depkg','dwy']
        self.manifest={'method_batch':{'methods':methods,'datasets':datasets,'seeds':[17],
            'repetition_policy':'single_run_first',
            'job_ids':[f'{m.lower()}_{d}_s17' for m in methods for d in datasets]}}

    def test_all_and_only_selected_groups_use_one_run(self):
        for method in ['LSMGA','DMKGC','IMKGC']:
            for dataset in ['dbp5l','depkg','dwy']:
                self.assertEqual(required_seeds(self.manifest,method,dataset),[17])
            self.assertEqual(required_seeds(self.manifest,method,'wk3l'),[17,29,43])
        self.assertEqual(required_seeds(self.manifest,'TransE','dbp5l'),[17,29,43])
        self.assertEqual(required_seeds({},'LSMGA','dwy'),[17,29,43])

    def test_single_run_has_no_standard_deviation(self):
        value,sd=summarize([0.2])
        self.assertEqual(value,0.2)
        self.assertIsNone(sd)
        value,sd=summarize([0.1,0.2,0.3])
        self.assertAlmostEqual(value,0.2)
        self.assertAlmostEqual(sd,0.1)

    def test_table_cells_require_only_the_selected_run(self):
        for dataset,job_dataset in [('dbp','dbp5l'),('epkg','depkg'),('dwy','dwy')]:
            for method in ['lsmga','dmkgc','imkgc']:
                for metric in ['mrr','h1','h10']:
                    spec=cell_scope(f'T2.{dataset}.{method}.{metric}',self.manifest)
                    self.assertEqual(spec['jobs'],[f'{method}_{job_dataset}_s17'])
        self.assertEqual(cell_scope('T2.dwy.qura.mrr',self.manifest)['jobs'],[])

    def test_cannot_silently_choose_an_unplanned_seed(self):
        self.manifest['method_batch']['seeds']=[29]
        with self.assertRaises(AssertionError):required_seeds(self.manifest,'IMKGC','dwy')


if __name__=='__main__':unittest.main()
