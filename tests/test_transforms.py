import doctest
import retnext.transforms

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(retnext.transforms))
    return tests
