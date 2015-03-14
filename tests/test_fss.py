#!/usr/bin/env python
# encoding: utf-8

# Copyright 2014 Max Planck Society, Andreas Sorge
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import inspect
import numpy as np
import scipy.optimize
import copy

import fssa


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
            hasattr(fssa, 'scaledata'),
            msg='No such function: fssa.scaledata'
        )

    def test_signature(self):
        """
        Test scaledata function signature
        """
        try:
            args = inspect.signature(fssa.scaledata).parameters
        except:
            args = inspect.getargspec(fssa.scaledata).args

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
        self._test_for_1d_array_like(fssa.scaledata, 'rho', self.default_params)

    def test_l_1d_array_like(self):
        """
        Test that function raises ValueError when l is not 1-D array_like
        """

        # l should be 1-D array_like
        self._test_for_1d_array_like(fssa.scaledata, 'l', self.default_params)

    def test_a_2d_array_like(self):
        """
        Test that function raises ValueError when a is not 2-D array_like
        """

        # a should be 2-D array_like
        self._test_for_2d_array_like(fssa.scaledata, 'a', self.default_params)

    def test_a_has_l_rho_shape(self):
        """
        Test that function raises ValueError when a does not have the correct
        shape
        """

        # a should have shape (rho.size, l.size)
        args = copy.deepcopy(self.default_params)
        args['a'] = np.ones(shape=(3, args['rho'].size - 1))
        self.assertRaises(ValueError, fssa.scaledata, **args)

    def test_da_2d_array_like(self):
        """
        Test that function raises ValueError when da is not 2-D array_like
        """

        # da should be 2-D array_like
        self._test_for_2d_array_like(fssa.scaledata, 'da', self.default_params)

    def test_da_has_l_rho_shape(self):
        """
        Test that function raises ValueError when da does not have the correct
        shape
        """

        # da should have shape (rho.size, l.size)
        args = copy.deepcopy(self.default_params)
        args['da'] = np.ones(shape=(3, args['rho'].size - 1))
        self.assertRaises(ValueError, fssa.scaledata, **args)

    def test_da_all_positive(self):
        """
        Test that function raises ValueError if any da is not positive
        """

        # da should have only positive entries
        args = copy.deepcopy(self.default_params)
        args['da'][1, 1] = 0.
        self.assertRaises(ValueError, fssa.scaledata, **args)

        args = copy.deepcopy(self.default_params)
        args['da'][1, 0] = -1.
        self.assertRaises(ValueError, fssa.scaledata, **args)

    def test_rho_c_float(self):
        """
        Test that function raises TypeError when rho_c is not a float
        """

        # rho_c should be a float
        self._test_for_float(fssa.scaledata, 'rho_c', self.default_params)

    def test_rho_c_in_range(self):
        """
        Test that function raises ValueError when rho_c is out of range
        """

        args = copy.deepcopy(self.default_params)
        args['rho_c'] = args['rho'].max() + 1.
        self.assertRaises(ValueError, fssa.scaledata, **args)

    def test_nu_float(self):
        """
        Test that function raises TypeError when nu is not a float
        """

        # nu should be a float
        self._test_for_float(fssa.scaledata, 'nu', self.default_params)

    def test_zeta_float(self):
        """
        Test that function raises TypeError when zeta is not a float
        """

        # zeta should be a float
        self._test_for_float(fssa.scaledata, 'zeta', self.default_params)

    def test_output_len(self):
        """
        Test that function returns at least three items
        """
        result = fssa.scaledata(**self.default_params)
        self.assertGreaterEqual(len(result), 3)

    def test_output_shape(self):
        """
        Test that function returns three ndarrays of correct shape
        """
        correct_shape = (
            self.default_params['l'].size,
            self.default_params['rho'].size,
        )
        result = fssa.scaledata(**self.default_params)
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
        result = fssa.scaledata(**self.default_params)
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
        result = fssa.scaledata(**self.default_params)
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

        result = fssa.scaledata(**self.default_params)
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
            hasattr(fssa.fss, '_jprimes'),
            msg='No such function: fssa.fss._jprimes'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fssa.fss._jprimes).parameters
        except:
            args = inspect.getargspec(fssa.fss._jprimes).args

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
        ret = fssa.fss._jprimes(x, 2)
        self.assertTupleEqual(ret.shape, x.shape)

    def test_never_select_from_i(self):
        """
        Test that the mask never selects from the i row
        """
        x = np.sort(np.random.rand(4, 3))
        ret = fssa.fss._jprimes(x, 2)
        self.assertTrue(np.isnan(ret[2, :]).all())

    def test_jprime_nan_if_xij_doesnt_fit(self):
        """
        Test that there is no j' if and only if the xij does not fit into
        the i' row of x values
        """
        x = np.sort(np.random.rand(4, 3))

        # degenerate the i row
        x[3, :] = x[2, :]

        ret = fssa.fss._jprimes(x, 2)

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

        ret = fssa.fss._jprimes(x, 2)

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

        ret = fssa.fss._jprimes(x, 2)

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
            hasattr(fssa.fss, '_select_mask'),
            msg='No such function: fssa.fss._select_mask'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fssa.fss._select_mask).parameters
        except:
            args = inspect.getargspec(fssa.fss._select_mask).args

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
        ret = fssa.fss._select_mask(**self.default_args)
        self.assertTupleEqual(
            ret.shape, self.default_args['j_primes'].shape
        )

    def test_jprime_selection(self):
        """
        Test that the function selects element (i', j') if and only if
        j_primes[i', j] == j' or j_primes[i', j] == j' - 1
        """
        ret = fssa.fss._select_mask(**self.default_args)
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
            hasattr(fssa.fss, '_wls_linearfit_predict'),
            msg='No such function: fssa.fss._wls_linearfit_predict'
        )

    def test_signature(self):
        """
        Test wls function signature
        """
        try:
            args = inspect.signature(fssa.fss._wls_linearfit_predict).parameters
        except:
            args = inspect.getargspec(fssa.fss._wls_linearfit_predict).args

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
        ret = fssa.fss._wls_linearfit_predict(**self.default_args)
        y = float(ret[0])
        dy = float(ret[1])
        y, dy = fssa.fss._wls_linearfit_predict(**self.default_args)
        float(y), float(dy)

    def test_function_value(self):
        """
        Test for a concrete function value
        """
        y, dy2 = fssa.fss._wls_linearfit_predict(**self.default_args)
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
        self.scaled_data = fssa.scaledata(l, rho, a, da, 0, 1, 0)

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fssa, 'quality'),
            msg='No such function: fssa.quality'
        )

    def test_signature(self):
        """
        Test quality function signature
        """
        try:
            args = inspect.signature(fssa.quality).parameters
        except:
            args = inspect.getargspec(fssa.quality).args

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
                        fssa.quality, i, self.scaled_data
                    )
            except:  # python 2
                self._test_for_2d_array_like(
                    fssa.quality, i, self.scaled_data
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
                    self.assertRaises(ValueError, fssa.quality, *args)
            except AttributeError:  # python 2
                self.assertRaises(ValueError, fssa.quality, *args)

    def test_x_sorted(self):
        """
        Test that function raises ValueError if x is not sorted in the second
        dimension (parameter values)
        """
        args = list(self.scaled_data)

        # manipulate x at some dimension
        args[0][1, 3] = args[0][1, 1]

        self.assertRaises(ValueError, fssa.quality, *args)

    def test_zero_quality(self):
        """
        Test that function returns close to zero value if fed with the same x
        and y values
        """
        def master_curve(x):
            return 1. / (1. + np.exp(5 * (1. - x)))

        x = np.linspace(0, 2)
        x_array = np.row_stack([x for i in range(10)])
        y_array = master_curve(x_array)
        dy = 0.05
        dy_array = dy * np.ones_like(y_array)
        ret = fssa.quality(x_array, y_array, dy_array)
        self.assertGreaterEqual(ret, 0.)
        self.assertLess(ret, 0.1)

    def test_standard_quality(self):
        """
        Test that function returns close to one value if fed with the same x
        and y values with some standard error applied
        """
        def master_curve(x):
            return 1. / (1. + np.exp(5 * (1. - x)))

        x = np.linspace(0, 2)
        x_array = np.row_stack([x for i in range(10)])
        y_array = master_curve(x_array)
        dy = 0.05
        dy_array = dy * np.ones_like(y_array)
        y_array += dy * np.random.randn(*y_array.shape)
        ret = fssa.quality(x_array, y_array, dy_array)
        self.assertGreater(ret, np.power(10, -0.5))
        self.assertLess(ret, np.power(10, 0.5))

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


class TestNelderMeadErrors(unittest.TestCase):
    """
    Test the Nelder Mead errors helper function
    """

    def setUp(self):
        self.n = 2
        self.sim = np.array([[0, 0], [1, 0], [0, 1]])
        self.ymin = 1.42

        def identity_curvature(x):
            return self.ymin + (
                0.5 * np.sum(x**2) if x.ndim == 1
                else 0.5 * np.sum(x**2, axis=1)
            )

        self.fun = identity_curvature
        self.fsim = self.fun(self.sim)
        self.default_args = {
            'sim': self.sim,
            'fun': self.fun,
            'fsim': self.fsim,
        }

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fssa.fss, '_neldermead_errors'),
            msg='No such function: fssa.fss._neldermead_errors'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fssa.fss._neldermead_errors).parameters
        except:
            args = inspect.getargspec(fssa.fss._neldermead_errors).args

        fields = ['sim', 'fsim', 'fun']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_return_errors_and_varco_matrix(self):
        """
        Test that the function returns the standard errors and the whole
        variance--covariance matrix
        """
        errors, varco = fssa.fss._neldermead_errors(**self.default_args)
        self.assertTupleEqual(errors.shape, (self.n, ))
        self.assertTupleEqual(varco.shape, (self.n, self.n))

    def test_identity_hessian(self):
        """
        Test function returns correct errors and variance--covariance matrix for
        identity hessian (curvature)
        """
        errors, varco = fssa.fss._neldermead_errors(**self.default_args)
        self.assertTrue(np.allclose(errors, np.sqrt(2. * self.ymin)))
        self.assertTrue(np.allclose(varco, 2. * self.ymin * np.eye(self.n)))

    def test_ellipsoidal_hessian(self):
        """
        Test function returns correct errors and variance--covariance matrix for
        ellipsoidal hessian (curvature)
        """
        args = copy.deepcopy(self.default_args)

        def ellipsoidal_curvature(x):
            return self.ymin + (
                0.5 * np.sum(x**2) - 0.5 * np.prod(x) if x.ndim == 1
                else 0.5 * np.sum(x**2, axis=1) - 0.5 * np.prod(x, axis=1)
            )

        args['fun'] = ellipsoidal_curvature
        errors, varco = fssa.fss._neldermead_errors(**args)
        self.assertTrue(np.allclose(errors, np.sqrt(2. * self.ymin * 4. / 3.)))
        self.assertTrue(np.allclose(
            varco,
            2. * self.ymin * np.eye(self.n) * 4. / 3.
            + (np.ones((self.n, self.n)) - np.eye(self.n)) * 2. * self.ymin
            * 2. / 3.
        ))


class TestNelderMead(unittest.TestCase):
    """
    Test the Nealder Mead with errors implementation
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fssa.fss, '_minimize_neldermead_witherrors'),
            msg='No such function: fssa.fss._minimize_neldermead_witherrors'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(
                fssa.fss._minimize_neldermead_witherrors
            ).parameters
        except:
            args = inspect.getargspec(
                fssa.fss._minimize_neldermead_witherrors
            ).args

        fields = ['fun', 'x0', 'args', 'callback', 'xtol', 'ftol', 'maxiter',
                  'maxfev', 'disp', 'return_all', 'with_errors'
                  ]

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_exact_results_as_original_method(self):
        """
        Test that the modified method returns exactly the same results as the
        original method (except for the additional errors)
        """
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        res_original = scipy.optimize.minimize(
            scipy.optimize.rosen, x0, method='Nelder-Mead'
        )

        res_witherrors = scipy.optimize.minimize(
            scipy.optimize.rosen,
            x0,
            method=fssa.fss._minimize_neldermead_witherrors
        )

        fields = ['fun', 'status', 'success', 'message']
        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertEqual(
                        res_original[field],
                        res_witherrors[field]
                    )
            except AttributeError:  # python 2
                self.assertEqual(
                    res_original[field],
                    res_witherrors[field]
                )

        self.assertTrue(np.all(
            res_original['x'] == res_witherrors['x']
        ))

    def test_returns_errors(self):
        """
        Test that the modified method returns the errors and
        variance--covariance matrix
        """
        x0 = [1.3, 0.7, 0.8, 1.9, 1.2]

        res = scipy.optimize.minimize(
            scipy.optimize.rosen,
            x0,
            method=fssa.fss._minimize_neldermead_witherrors
        )

        errors, varco = fssa.fss._neldermead_errors(
            sim=res['sim'],
            fsim=res['fsim'],
            fun=scipy.optimize.rosen
        )

        self.assertTrue(np.all(res['errors'] == errors))
        self.assertTrue(np.all(res['varco'] == varco))


class TestAutoscale(unittest.TestCase):
    """
    Test the autoscale function
    """

    def setUp(self):
        self.l = [ 10, 100, 1000 ]
        self.rho = np.linspace(0.9, 1.1)

        l_mesh, rho_mesh = np.meshgrid(self.l, self.rho, indexing='ij')
        self.x = np.power(l_mesh, 0.5) * (rho_mesh - 1.)

        def master_curve(x):
            return 1. / (1. + np.exp( - x))

        self.master_curve = master_curve

        self.y = self.master_curve(self.x)
        self.y += np.random.randn(*self.y.shape) * self.y / 100.
        self.dy = self.y / 100.
        self.a = self.y
        self.da = self.dy
        self.rho_c0 = 0.95
        self.nu0 = 2.0
        self.zeta0 = 0.0
        self.rho_c = 1.0
        self.nu = 2.0
        self.zeta = 0.0

        self.default_args = {
            'l': self.l,
            'rho': self.rho,
            'a': self.a,
            'da': self.da,
            'rho_c0': self.rho_c0,
            'nu0': self.nu0,
            'zeta0': self.zeta0
        }

    def tearDown(self):
        pass

    def test_call(self):
        """
        Test function call
        """
        self.assertTrue(
            hasattr(fssa, 'autoscale'),
            msg='No such function: fssa.autoscale'
        )

    def test_signature(self):
        """
        Test function signature
        """
        try:
            args = inspect.signature(fssa.autoscale).parameters
        except:
            args = inspect.getargspec(fssa.autoscale).args

        fields = ['l', 'rho', 'a', 'da', 'rho_c0', 'nu0', 'zeta0']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, args)
            except AttributeError:  # python 2
                self.assertIn(field, args)

    def test_return_type(self):
        """
        Test that function returns correct type
        """
        res = fssa.autoscale(**self.default_args)

        fields = ['success', 'x', 'rho', 'nu', 'zeta', 'drho', 'dnu', 'dzeta',
                  'errors', 'varco']

        for field in fields:
            try:  # python 3
                with self.subTest(i=field):
                    self.assertIn(field, res)
            except AttributeError:  # python 2
                self.assertIn(field, res)
