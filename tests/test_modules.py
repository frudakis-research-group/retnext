import doctest
import retnext.modules

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(retnext.modules))
    return tests
