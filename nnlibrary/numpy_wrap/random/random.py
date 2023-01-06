from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from nnlibrary.numpy_wrap.node import AbstractNode
from nnlibrary.numpy_wrap.node import Node
from nnlibrary.numpy_wrap.node import node_utils


def rand(*dn: Any) -> AbstractNode:
    dn = [node_utils.get_values_if_needed(el) for el in dn]
    values = npw.numpy.random.rand(*dn)
    return Node(values=values)  # partials would be zeros


def randint(low: Union[int, npw.ndarray, Iterable, float, AbstractNode],
            high: Union[int, npw.ndarray, Iterable, float, None, AbstractNode] = None,
            size: Union[int, Iterable, tuple[int], None] = None,
            dtype: Optional[object] = None) -> AbstractNode:
    low = node_utils.get_values_if_needed(x=low)
    high = node_utils.get_values_if_needed(x=high)
    values = npw.numpy.random.randint(low=low, high=high, size=size, dtype=dtype)
    return Node(values=values)  # partials would be zeros


def randn(*dn: Any) -> AbstractNode:
    dn = [node_utils.get_values_if_needed(el) for el in dn]
    values = npw.numpy.random.randn(*dn)
    return Node(values=values)  # partials would be zeros


def random(size: Any = None) -> AbstractNode:
    values = npw.numpy.random.random(size=size)
    return Node(values=values)  # partials would be zeros


def random_integers(low: Union[int, AbstractNode],
                    high: Optional[Union[int, AbstractNode]] = None,
                    size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    low = node_utils.get_values_if_needed(x=low)
    high = node_utils.get_values_if_needed(x=high)
    values = npw.numpy.random.random_integers(low=low, high=high, size=size)
    return Node(values=values)  # partials would be zeros


def random_sample(size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    values = npw.numpy.random.random_sample(size=size)
    return Node(values=values)  # partials would be zeros


def sample(*args: Any, **kwargs: Any) -> AbstractNode:
    values = npw.numpy.random.sample(*args, **kwargs)
    return Node(values=values)  # partials would be zeros


def ranf(*args: Any, **kwargs: Any) -> AbstractNode:
    values = npw.numpy.random.ranf(*args, **kwargs)
    return Node(values=values)  # partials would be zeros


def choice(a: Union[Any, AbstractNode],
           size: Union[int, Iterable, tuple[int], None] = None,
           replace: Optional[bool] = True,
           p: Union[Any, AbstractNode] = None) -> AbstractNode:
    p = node_utils.get_values_if_needed(x=p)
    a = node_utils.get_values_if_needed(x=a)
    idx = npw.numpy.random.choice(a=a.size, size=size, replace=replace, p=p)
    return Node(values=a[idx], partials=a.partials[idx])


def permutation(x: Union[int, npw.ndarray, Iterable, float, AbstractNode]) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    idx = npw.numpy.random.permutation(x.values.shape[0])
    return Node(values=x.values[idx], partials=x.partials[idx])


def shuffle(x: Union[int, npw.ndarray, Iterable, float, AbstractNode]) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    idx = npw.numpy.random.permutation(x.values.shape[0])
    x.values, x.partials = x.values[idx], x.partials[idx]
    return x


def beta(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         b: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    b = node_utils.get_values_if_needed(x=b)
    values = npw.numpy.random.beta(a=a, b=b, size=size)
    return Node(values=values)  # partials would be zeros


def binomial(n: Union[int, npw.ndarray, Iterable, float, AbstractNode],
             p: Union[float, npw.ndarray, Iterable, int, AbstractNode],
             size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    n = node_utils.get_values_if_needed(x=n)
    p = node_utils.get_values_if_needed(x=p)
    values = npw.numpy.random.binomial(n=n, p=p, size=size)
    return Node(values=values)  # partials would be zeros


def chisquare(df: Union[float, npw.ndarray, Iterable, int, AbstractNode],
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    df = node_utils.get_values_if_needed(x=df)
    values = npw.numpy.random.chisquare(df=df, size=size)
    return Node(values=values)  # partials would be zeros


def dirichlet(alpha: Union[Iterable, AbstractNode],
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    alpha = node_utils.get_values_if_needed(x=alpha)
    values = npw.numpy.random.dirichlet(alpha=alpha, size=size)
    return Node(values=values)  # partials would be zeros


def exponential(scale: Union[float, npw.ndarray, Iterable, int, AbstractNode] = 1.0,
                size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.exponential(scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def f(dfnum: Union[float, npw.ndarray, Iterable, int, AbstractNode],
      dfden: Union[float, npw.ndarray, Iterable, int, float, AbstractNode],
      size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    dfnum = node_utils.get_values_if_needed(x=dfnum)
    dfden = node_utils.get_values_if_needed(x=dfden)
    values = npw.numpy.random.f(dfnum=dfnum, dfden=dfden, size=size)
    return Node(values=values)  # partials would be zeros


def gamma(shape: Union[float, npw.ndarray, Iterable, int, AbstractNode],
          scale: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
          size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    shape = node_utils.get_values_if_needed(x=shape)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.gamma(shape=shape, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def geometric(p: Union[float, npw.ndarray, Iterable, int, AbstractNode],
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    p = node_utils.get_values_if_needed(x=p)
    values = npw.numpy.random.geometric(p=p, size=size)
    return Node(values=values)  # partials would be zeros


def gumbel(loc: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 0.0,
           scale: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
           size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    loc = node_utils.get_values_if_needed(x=loc)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.gumbel(loc=loc, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def hypergeometric(ngood: Union[int, npw.ndarray, Iterable, float, AbstractNode],
                   nbad: Union[int, npw.ndarray, Iterable, float, AbstractNode],
                   nsample: Union[int, npw.ndarray, Iterable, float, AbstractNode],
                   size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    ngood = node_utils.get_values_if_needed(x=ngood)
    nbad = node_utils.get_values_if_needed(x=nbad)
    nsample = node_utils.get_values_if_needed(x=nsample)
    values = npw.numpy.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)
    return Node(values=values)  # partials would be zeros


def laplace(loc: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 0.0,
            scale: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
            size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    loc = node_utils.get_values_if_needed(x=loc)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.laplace(loc=loc, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def logistic(loc: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 0.0,
             scale: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
             size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    loc = node_utils.get_values_if_needed(x=loc)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.logistic(loc=loc, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def lognormal(mean: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 0.0,
              sigma: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    mean = node_utils.get_values_if_needed(x=mean)
    sigma = node_utils.get_values_if_needed(x=sigma)
    values = npw.numpy.random.lognormal(mean=mean, sigma=sigma, size=size)
    return Node(values=values)  # partials would be zeros


def logseries(p: Union[float, npw.ndarray, Iterable, int, AbstractNode],
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    p = node_utils.get_values_if_needed(x=p)
    values = npw.numpy.random.logseries(p=p, size=size)
    return Node(values=values)  # partials would be zeros


def multinomial(n: Union[int, AbstractNode],
                pvals: Union[Iterable, AbstractNode],
                size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    n = node_utils.get_values_if_needed(x=n)
    pvals = node_utils.get_values_if_needed(x=pvals)
    values = npw.numpy.random.multinomial(n=n, pvals=pvals, size=size)
    return Node(values=values)  # partials would be zeros


def multivariate_normal(mean: Any,
                        cov: Any,
                        size: Union[int, Iterable, tuple[int], None] = None,
                        check_valid: Optional[str] = 'warn',
                        tol: Optional[Union[float, AbstractNode]] = 1,
                        *args: Any,
                        **kwargs: Any) -> AbstractNode:
    mean = node_utils.get_values_if_needed(x=mean)
    cov = node_utils.get_values_if_needed(x=cov)
    tol = node_utils.get_values_if_needed(x=tol)
    values = npw.numpy.random.multivariate_normal(mean=mean, cov=cov, size=size,
                                                  check_valid=check_valid, tol=tol, *args, **kwargs)
    return Node(values=values)  # partials would be zeros


def negative_binomial(n: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                      p: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                      size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    n = node_utils.get_values_if_needed(x=n)
    p = node_utils.get_values_if_needed(x=p)
    values = npw.numpy.random.negative_binomial(n=n, p=p, size=size)
    return Node(values=values)  # partials would be zeros


def noncentral_chisquare(df: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                         nonc: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                         size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    df = node_utils.get_values_if_needed(x=df)
    nonc = node_utils.get_values_if_needed(x=nonc)
    values = npw.numpy.random.noncentral_chisquare(df=df, nonc=nonc, size=size)
    return Node(values=values)  # partials would be zeros


def noncentral_f(dfnum: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                 dfden: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                 nonc: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                 size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    dfnum = node_utils.get_values_if_needed(x=dfnum)
    dfden = node_utils.get_values_if_needed(x=dfden)
    nonc = node_utils.get_values_if_needed(x=nonc)
    values = npw.numpy.random.noncentral_f(dfnum=dfnum, dfden=dfden, nonc=nonc, size=size)
    return Node(values=values)  # partials would be zeros


def normal(loc: Union[float, npw.ndarray, Iterable, int, AbstractNode] = 0.0,
           scale: Union[float, npw.ndarray, Iterable, int, AbstractNode] = 1.0,
           size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    loc = node_utils.get_values_if_needed(x=loc)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.normal(loc=loc, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def pareto(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
           size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    values = npw.numpy.random.pareto(a=a, size=size)
    return Node(values=values)  # partials would be zeros


def poisson(lam: Union[float, npw.ndarray, Iterable, int, AbstractNode] = 1.0,
            size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    lam = node_utils.get_values_if_needed(x=lam)
    values = npw.numpy.random.poisson(lam=lam, size=size)
    return Node(values=values)  # partials would be zeros


def power(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
          size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    values = npw.numpy.random.power(a=a, size=size)
    return Node(values=values)  # partials would be zeros


def rayleigh(scale: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 1.0,
             size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.rayleigh(scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def standard_cauchy(size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    values = npw.numpy.random.standard_cauchy(size=size)
    return Node(values=values)  # partials would be zeros


def standard_exponential(size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    values = npw.numpy.random.standard_exponential(size=size)
    return Node(values=values)  # partials would be zeros


def standard_gamma(shape: Union[float, npw.ndarray, Iterable, int, AbstractNode],
                   size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    shape = node_utils.get_values_if_needed(x=shape)
    values = npw.numpy.random.standard_gamma(shape=shape, size=size)
    return Node(values=values)  # partials would be zeros


def standard_normal(size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    values = npw.numpy.random.standard_normal(size=size)
    return Node(values=values)  # partials would be zeros


def standard_t(df: Union[float, npw.ndarray, Iterable, int, AbstractNode],
               size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    df = node_utils.get_values_if_needed(x=df)
    values = npw.numpy.random.standard_t(df=df, size=size)
    return Node(values=values)  # partials would be zeros


def triangular(left: Union[float, npw.ndarray, Iterable, int, AbstractNode],
               mode: Union[float, npw.ndarray, Iterable, int, AbstractNode],
               right: Union[float, npw.ndarray, Iterable, int, AbstractNode],
               size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    left = node_utils.get_values_if_needed(x=left)
    mode = node_utils.get_values_if_needed(x=mode)
    right = node_utils.get_values_if_needed(x=right)
    values = npw.numpy.random.triangular(left=left, mode=mode, right=right, size=size)
    return Node(values=values)  # partials would be zeros


def uniform(low: Union[float, npw.ndarray, Iterable, int, None, AbstractNode] = 0.0,
            high: Union[float, npw.ndarray, Iterable, int, AbstractNode] = 1.0,
            size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    low = node_utils.get_values_if_needed(x=low)
    high = node_utils.get_values_if_needed(x=high)
    values = npw.numpy.random.uniform(low=low, high=high, size=size)
    return Node(values=values)  # partials would be zeros


def vonmises(mu: Union[float, npw.ndarray, Iterable, int, AbstractNode],
             kappa: Union[float, npw.ndarray, Iterable, int, AbstractNode],
             size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    mu = node_utils.get_values_if_needed(x=mu)
    kappa = node_utils.get_values_if_needed(x=kappa)
    values = npw.numpy.random.vonmises(mu=mu, kappa=kappa, size=size)
    return Node(values=values)  # partials would be zeros


def wald(mean: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         scale: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    mean = node_utils.get_values_if_needed(x=mean)
    scale = node_utils.get_values_if_needed(x=scale)
    values = npw.numpy.random.wald(mean=mean, scale=scale, size=size)
    return Node(values=values)  # partials would be zeros


def weibull(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
            size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    values = npw.numpy.random.weibull(a=a, size=size)
    return Node(values=values)  # partials would be zeros


def zipf(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    values = npw.numpy.random.zipf(a=a, size=size)
    return Node(values=values)  # partials would be zeros
