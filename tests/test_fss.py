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
        l = np.logspace(1, 3, num=3)
        l_mesh, rho_mesh = np.meshgrid(l, rho, indexing='ij')
        a = 1. / (1. + np.exp(- np.log10(l_mesh) * rho_mesh))
        da = np.ones_like(a) * 1e-2

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

    def test_values(self):
        """
        Check some arbitrary value
        """

        l = self.default_params['l'][2]
        rho = self.default_params['rho'][3]
        rho_c = self.default_params['rho_c']
        nu = self.default_params['nu']
        zeta = self.default_params['zeta']
        a = self.default_params['a'][2, 3]
        da = self.default_params['da'][2, 3]

        result = fss.scaledata(**self.default_params)
        self.assertAlmostEqual(result.x[2, 3], l ** (1. / nu) * (rho - rho_c))
        self.assertAlmostEqual(result.y[2, 3], l ** (- zeta / nu) * a)
        self.assertAlmostEqual(result.dy[2, 3], l ** (- zeta / nu) * da)

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


class TestJPrimes(unittest.TestCase):
    """
    Test the j_primes helper function
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_existence(self):
        """
        Test for function existence
        """
        self.assertTrue(
            hasattr(fss.fss, '_jprimes'),
            msg='No such function: fss.fss._jprimes'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fss.fss._jprimes).parameters
        except:
            args = inspect.getargspec(fss.fss._jprimes).args

        fields = ['x', 'i']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_return_type(self):
        """
        Test that the function returns an array of the same shape as x
        """
        x = np.sort(np.random.rand(4, 3))
        ret = fss.fss._jprimes(x, 2)
        self.assertTupleEqual(ret.shape, x.shape)

    def test_never_select_from_i(self):
        """
        Test that the mask never selects from the i row
        """
        x = np.sort(np.random.rand(4, 3))
        ret = fss.fss._jprimes(x, 2)
        self.assertTrue(np.isnan(ret[2, :]).all())

    def test_jprime_nan_if_xij_doesnt_fit(self):
        """
        Test that there is no j' if and only if the xij does not fit into
        the i' row of x values
        """
        x = np.sort(np.random.rand(4, 3))

        # degenerate the i row
        x[3, :] = x[2, :]

        ret = fss.fss._jprimes(x, 2)

        for iprime in range(x.shape[0]):
            if iprime == 2:
                continue
            for j in range(x.shape[1]):
                xij = x[2, j]
                jprime = ret[iprime, j]
                self.assertTrue(
                    np.isnan(jprime) ==
                    (
                        xij < x[iprime, :].min()
                        or xij >= x[iprime, :].max()
                    )
                )

    def test_xiprimejprime_lessequal_xij(self):
        """
        Test that :math:`x_{i'j'} \leq x_{ij}`
        """
        x = np.sort(np.random.rand(4, 3))

        # degenerate the i row
        x[3, :] = x[2, :]

        ret = fss.fss._jprimes(x, 2)

        for iprime in range(x.shape[0]):
            if iprime == 2:
                continue
            for j in range(x.shape[1]):
                xij = x[2, j]
                jprime = ret[iprime, j]
                if np.isnan(jprime):
                    continue
                jprime = int(jprime)
                self.assertLessEqual(x[iprime, jprime], xij)

    def test_xiprimejprime1_greater_xij(self):
        """
        Test that :math:`x_{i'(j'+1)} > x_{ij}`
        """
        x = np.sort(np.random.rand(4, 3))

        # degenerate the i row
        x[3, :] = x[2, :]

        ret = fss.fss._jprimes(x, 2)

        for iprime in range(x.shape[0]):
            if iprime == 2:
                continue
            for j in range(x.shape[1]):
                xij = x[2, j]
                jprime = ret[iprime, j]
                if np.isnan(jprime):
                    continue
                jprime = int(jprime)
                self.assertGreater(x[iprime, jprime + 1], xij)


class TestSelectMask(unittest.TestCase):
    """
    Test the select mask helper function
    """

    def setUp(self):
        self.i = 1
        self.j = 2
        self.j_primes = np.asfarray(np.random.randint(-1, 2, size=(4, 3)))
        self.j_primes[self.j_primes == -1] = np.nan
        self.default_args = {
            'j': self.j,
            'j_primes': self.j_primes,
        }

    def tearDown(self):
        pass

    def test_existence(self):
        """
        Test for function existence
        """
        self.assertTrue(
            hasattr(fss.fss, '_select_mask'),
            msg='No such function: fss.fss._select_mask'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fss.fss._select_mask).parameters
        except:
            args = inspect.getargspec(fss.fss._select_mask).args

        fields = ['j', 'j_primes']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_return_type(self):
        """
        Test that the function returns an array of the same shape as
        j_primes
        """
        ret = fss.fss._select_mask(**self.default_args)
        self.assertTupleEqual(
            ret.shape, self.default_args['j_primes'].shape
        )

    def test_jprime_selection(self):
        """
        Test that the function selects element (i', j') if and only if
        j_primes[i', j] == j' or j_primes[i', j] == j' - 1
        """
        ret = fss.fss._select_mask(**self.default_args)
        for iprime in range(self.j_primes.shape[0]):
            for jprime in range(self.j_primes.shape[1]):
                self.assertTrue(
                    ret[iprime, jprime] ==
                    (
                        (self.j_primes[iprime, self.j] == jprime)
                        or
                        (self.j_primes[iprime, self.j] == jprime - 1)
                    )
                )


class TestWLSPredict(unittest.TestCase):
    """
    Test the Weighted Least Squares Prediction Function
    """

    def setUp(self):
        # set up default arguments
        x = np.array([-2, 0, -10, 10], dtype=float)
        y = np.array([-4, 0, -100, 100], dtype=float)
        dy = np.array([0.2, 0.1, 1, 1])
        w = dy ** (-2)
        wx = w * x
        wy = w * y
        wxx = w * x * x
        wxy = w * x * y
        select = np.ones_like(x, dtype=bool)
        x = 1.0
        self.default_args = {
            'x': x,
            'w': w,
            'wx': wx,
            'wy': wy,
            'wxx': wxx,
            'wxy': wxy,
            'select': select,
        }

    def tearDown(self):
        pass

    def test_existence(self):
        """
        Test for function existence
        """
        self.assertTrue(
            hasattr(fss.fss, '_wls_linearfit_predict'),
            msg='No such function: fss.fss._wls_linearfit_predict'
        )

    def test_signature(self):
        """
        Test wls function signature
        """
        try:
            args = inspect.signature(fss.fss._wls_linearfit_predict).parameters
        except:
            args = inspect.getargspec(fss.fss._wls_linearfit_predict).args

        fields = ['x', 'w', 'wx', 'wy', 'wxx', 'wxy', 'select']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_return_type(self):
        """
        Tests for correct return
        """
        ret = fss.fss._wls_linearfit_predict(**self.default_args)
        y = float(ret[0])
        dy = float(ret[1])
        y, dy = fss.fss._wls_linearfit_predict(**self.default_args)
        float(y), float(dy)

    def test_function_value(self):
        """
        Test for a concrete function value
        """
        y, dy2 = fss.fss._wls_linearfit_predict(**self.default_args)
        self.assertAlmostEqual(
            y, float(274400. / 35600. + 80000. / 35600.)
        )
        self.assertAlmostEqual(
            dy2, 527. / 35600.
        )


class TestQuality(unittest.TestCase):
    """
    Test the quality function
    """

    def setUp(self):
        rho = np.linspace(-1, 1, num=11)
        l = np.logspace(1, 3, num=3)
        l_mesh, rho_mesh = np.meshgrid(l, rho, indexing='ij')
        a = 1. / (1. + np.exp(- np.log10(l_mesh) * rho_mesh))
        da = np.ones_like(a) * 1e-2
        self.scaled_data = fss.scaledata(l, rho, a, da, 0, 1, 0)

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fss, 'quality'),
            msg='No such function: fss.quality'
        )

    def test_signature(self):
        """
        Test quality function signature
        """
        try:
            args = inspect.signature(fss.quality).parameters
        except:
            args = inspect.getargspec(fss.quality).args

        fields = ['x', 'y', 'dy']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_args_2d_array_like(self):
        """
        Test that function raises ValueError when one of the arguments is not
        2-D array_like
        """

        # a should be 2-D array_like
        args = ['x', 'y', 'dy']

        for i in range(len(args)):
            try:  # python 3
                with self.subTest(i=i):
                    self._test_for_2d_array_like(
                        fss.quality, i, self.scaled_data
                    )
            except:  # python 2
                self._test_for_2d_array_like(
                    fss.quality, i, self.scaled_data
                )

    def test_args_of_same_shape(self):
        """
        Test that function raises ValueError if the shapes of the arguments
        differ
        """
        args = list(self.scaled_data)
        for i in range(len(args)):
            args = list(self.scaled_data)
            args[i] = np.zeros(shape=(134, 23))
            try:  # python 3
                with self.subTest(i=i):
                    self.assertRaises(ValueError, fss.quality, *args)
            except AttributeError:  # python 2
                self.assertRaises(ValueError, fss.quality, *args)

    def test_x_sorted(self):
        """
        Test that function raises ValueError if x is not sorted in the second
        dimension (parameter values)
        """
        args = list(self.scaled_data)

        # manipulate x at some dimension
        args[0][1, 3] = args[0][1, 1]

        self.assertRaises(ValueError, fss.quality, *args)

    def _test_for_2d_array_like(self, callable, param, default_args):
        """
        Test that callable raises ValueError when param is not 2-D array_like
        """
        args = list(default_args)
        args[param] = 0.0
        self.assertRaises(ValueError, callable, *args)

        args = list(default_args)
        args[param] = np.zeros(2)
        self.assertRaises(ValueError, callable, *args)

        args = list(default_args)
        args[param] = np.zeros(shape=(2, 3, 4))
        self.assertRaises(ValueError, callable, *args)

if __name__ == '__main__':
    unittest.main()
