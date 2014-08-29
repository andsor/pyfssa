#!/usr/bin/env python
# encoding: utf-8

import unittest
import inspect
import numpy as np
import copy

import fss


class TestScaleData(unittest.TestCase):

    """Test data scaling function."""

    def setUp(self):
        rho = np.linspace(-1, 1, num=11)
        l   = np.logspace(1, 3, num=3)
        l_mesh, rho_mesh = np.meshgrid(l, rho, indexing='ij')
        a   = 1. / (1. + np.exp(- np.log10(l_mesh) * rho_mesh))
        da  = np.ones_like(a) * 1e-2

        self.default_params = {
            'rho':   rho,
            'l':     l,
            'a':     a,
            'da':    da,
            'rho_c': 0.,
            'nu':    1.,
            'zeta':  0.,
        }

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fss, 'scaledata'),
            msg='No such function: fss.scaledata'
        )

    def test_signature(self):
        """
        Test scaledata function signature
        """
        try:
            args = inspect.signature(fss.scaledata).parameters
        except:
            args = inspect.getargspec(fss.scaledata).args

        self.assertIn('rho', args)
        self.assertIn('l', args)
        self.assertIn('a', args)
        self.assertIn('da', args)
        self.assertIn('rho_c', args)
        self.assertIn('nu', args)
        self.assertIn('zeta', args)

    def test_rho_1d_array_like(self):
        """
        Test that function raises ValueError when rho is not 1-D array_like
        """

        # rho should be 1-D array_like
        self._test_for_1d_array_like(fss.scaledata, 'rho', self.default_params)

    def test_l_1d_array_like(self):
        """
        Test that function raises ValueError when l is not 1-D array_like
        """

        # l should be 1-D array_like
        self._test_for_1d_array_like(fss.scaledata, 'l', self.default_params)

    def test_a_2d_array_like(self):
        """
        Test that function raises ValueError when a is not 2-D array_like
        """

        # a should be 2-D array_like
        self._test_for_2d_array_like(fss.scaledata, 'a', self.default_params)

    def test_a_has_l_rho_shape(self):
        """
        Test that function raises ValueError when a does not have the correct
        shape
        """

        # a should have shape (rho.size, l.size)
        args = copy.deepcopy(self.default_params)
        args['a'] = np.ones(shape=(3, args['rho'].size - 1))
        self.assertRaises(ValueError, fss.scaledata, **args)

    def test_da_2d_array_like(self):
        """
        Test that function raises ValueError when da is not 2-D array_like
        """

        # da should be 2-D array_like
        self._test_for_2d_array_like(fss.scaledata, 'da', self.default_params)

    def test_da_has_l_rho_shape(self):
        """
        Test that function raises ValueError when da does not have the correct
        shape
        """

        # da should have shape (rho.size, l.size)
        args = copy.deepcopy(self.default_params)
        args['da'] = np.ones(shape=(3, args['rho'].size - 1))
        self.assertRaises(ValueError, fss.scaledata, **args)

    def test_da_all_positive(self):
        """
        Test that function raises ValueError if any da is not positive
        """

        # da should have only positive entries
        args = copy.deepcopy(self.default_params)
        args['da'][1, 1] = 0.
        self.assertRaises(ValueError, fss.scaledata, **args)

        args = copy.deepcopy(self.default_params)
        args['da'][1, 0] = -1.
        self.assertRaises(ValueError, fss.scaledata, **args)

    def test_rho_c_float(self):
        """
        Test that function raises TypeError when rho_c is not a float
        """

        # rho_c should be a float
        self._test_for_float(fss.scaledata, 'rho_c', self.default_params)

    def test_rho_c_in_range(self):
        """
        Test that function raises ValueError when rho_c is out of range
        """

        args = copy.deepcopy(self.default_params)
        args['rho_c'] = args['rho'].max() + 1.
        self.assertRaises(ValueError, fss.scaledata, **args)

    def test_nu_float(self):
        """
        Test that function raises TypeError when nu is not a float
        """

        # nu should be a float
        self._test_for_float(fss.scaledata, 'nu', self.default_params)

    def test_zeta_float(self):
        """
        Test that function raises TypeError when zeta is not a float
        """

        # zeta should be a float
        self._test_for_float(fss.scaledata, 'zeta', self.default_params)

    def test_output_len(self):
        """
        Test that function returns at least three items
        """
        result = fss.scaledata(**self.default_params)
        self.assertGreaterEqual(len(result), 3)

    def test_output_shape(self):
        """
        Test that function returns three ndarrays of correct shape
        """
        correct_shape = (
            self.default_params['l'].size,
            self.default_params['rho'].size,
        )
        result = fss.scaledata(**self.default_params)
        for i in range(3):
            try:  # python 3
                with self.subTest(i=i):
                    self.assertTupleEqual(
                        np.asarray(result[i]).shape,
                        correct_shape
                    )
            except AttributeError:  # python 2
                self.assertTupleEqual(
                    np.asarray(result[i]).shape, correct_shape
                )

    def test_output_namedtuple(self):
        """
        Test that function returns namedtuple with correct field names
        """
        result = fss.scaledata(**self.default_params)
        fields = ['x', 'y', 'dy']
        for i in range(len(fields)):
            try:  # python 3
                with self.subTest(i=i):
                    self.assertTrue(hasattr(result, fields[i]))
                    self.assertIs(getattr(result, fields[i]), result[i])
            except AttributeError:  # python 2
                self.assertTrue(hasattr(result, fields[i]))
                self.assertIs(getattr(result, fields[i]), result[i])

    def test_zero_zeta_keeps_y_values(self):
        """
        Check that $\zeta = 0$ does not affect the y values
        """
        args = copy.deepcopy(self.default_params)
        args['zeta'] = 0.0
        result = fss.scaledata(**self.default_params)
        self.assertTrue(np.all(args['a'] == result.y))

    def _test_for_float(self, callable, param, default_args):
        """
        Test that callable raises TypeError when param is not float
        """
        args = copy.deepcopy(default_args)
        args[param] = np.array([1., 2.])
        self.assertRaises(TypeError, callable, **args)

    def _test_for_1d_array_like(self, callable, param, default_args):
        """
        Test that callable raises ValueError when param is not 1-D array_like
        """
        args = copy.deepcopy(default_args)
        args[param] = 0.0
        self.assertRaises(ValueError, callable, **args)

        args = copy.deepcopy(default_args)
        args[param] = np.zeros(shape=(2, 3))
        self.assertRaises(ValueError, callable, **args)

    def _test_for_2d_array_like(self, callable, param, default_args):
        """
        Test that callable raises ValueError when param is not 2-D array_like
        """
        args = copy.deepcopy(default_args)
        args[param] = 0.0
        self.assertRaises(ValueError, callable, **args)

        args = copy.deepcopy(default_args)
        args[param] = np.zeros(2)
        self.assertRaises(ValueError, callable, **args)

        args = copy.deepcopy(default_args)
        args[param] = np.zeros(shape=(2, 3, 4))
        self.assertRaises(ValueError, callable, **args)


if __name__ == '__main__':
    unittest.main()
