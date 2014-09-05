.. _neldermead:

The Nelder--Mead algorithm
==========================

The Nelder--Mead algorithm :cite:`Nelder1965Simplex` attempts to minimize a
goal function :math:`f : \mathbb{R}^n \to \mathbb{R}` of an unconstrained
optimization problem.
As it only evaluates function values, but no derivatives, the Nelder--Mead
algorithm is a *direct search method* :cite:`Kolda2003Optimization`.
Although the method generally lacks rigorous convergence properties
:cite:`Lagarias1998Convergence,Price2002Convergent`, in practice the first few
iterations often yield satisfactory results :cite:`Singer2009NelderMead`.
Typically, each iteration evaluates the goal function only once or twice
:cite:`Singer2004Efficient`, which is why the Nelder--Mead algorithm is
comparatively fast if goal function evaluation is the computational bottleneck
:cite:`Singer2009NelderMead`.

The algorithm
-------------

Nelder & Mead :cite:`Nelder1965Simplex` refined a simplex method by Spendley et al. 
:cite:`Spendley1962Sequential`.
A simplex is the generalization of triangles in :math:`\mathbb{R}^2` to
:math:`n` dimensions: in :math:`\mathbb{R}^n`, a simplex is the convex hull of
:math:`n+1` vertices :math:`x_0, \ldots, x_n \in \mathbb{R}^n`.
Starting with an initial simplex, the algorithm attempts to decrease the
function values :math:`f_i := f(x_i)` at the vertices by a sequence of
elementary transformations of the simplex along the local landscape.
The algorithm *succeeds* when the simplex is sufficiently small (*domain
convergence test*), and/or when the function values :math:`f_i` are
sufficiently close (*function-value convergence test*).
The algorithm *fails* when it did not succeed after a given number of
iterations or function evaluations.
See Singer & Nead :cite:`Singer2009NelderMead` and references therein for a
complete description of the algorithm and the simplex transformations.

Uncertainties in parameter estimation
-------------------------------------

For parameter estimation, Spendley et al. :cite:`Spendley1962Sequential` and
Nelder & Mead :cite:`Nelder1965Simplex` provide a method to estimate the
uncertainties.
Fitting a quadratic surface to the vertices and the midpoints of the edges of
the final simplex yields an estimate for the variance--covariance matrix.
The variance--covariance matrix is :math:`\mathbf{Q} \mathbf{B}^{-1}
\mathbf{Q}^T` as originally given by Nelder & Mead :cite:`Nelder1965Simplex`,
despite the erratum on the original paper.
The errors are the square roots of the diagonal terms
:cite:`Bevington2003Data`.

Implementation
--------------

Scientific Python :cite:`Jones2001SciPy,Oliphant2007Python` implements the
Nelder--Mead method for the :py:func:`scipy.optimize.minimize` function.
Note that this implementation only returns the vertex with the lowest function
value, but not the whole final simplex.

