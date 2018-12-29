import numpy as _np

__all__ = [
    'add',
    'sum',
    'prod',
    'divide',
    'pow',
    'sq',
    'euclid',
    'gen_pow_matrix',
    'linsolve',
    'minpoly',
    'polyadd',
    'polyprod',
    'polydiv',
    'polyval',
    'deg',
    'isnull',
    'null_poly',
    'binpolyadd'
    'binpolyprod',
    'binpolydiv',
    'polyshift'
    '_floorlog2',
    '_ljust0',
]

_floorlog2 = lambda x: int(_np.floor(_np.log2(x))) if x != 0 else 0
_floorlog2.__doc__ = "количество значимых битов в дв. записи неотрицательного числа"

binpoly_arr2num = lambda arr: int("".join(map(str, arr)), base=2)
binpoly_num2arr = lambda num: _np.array(list(map(int, _np.binary_repr(num))))

polydeg = lambda arr: len(arr) - 1
polydeg.__doc__ = "степень многочлена (над любым полем), представленного массивом коэфф-тов"
isnull = lambda arr: arr[0] == 0
isnull.__doc__ = "проверка многочлена на равенство нулю"
null_poly = lambda: _np.array([0], dtype=int)
null_poly.__doc__ = "возвратить нулевой многочлен"

_ljust0 = lambda arr, n: _np.pad(arr, (n - len(arr), 0), 'constant', constant_values=0)
_ljust0.__doc__ = "дополняет массив arr нулями слева до длины n (если длина массива < n, ошибка)"
polyshift = lambda arr, n: _np.pad(arr, (0, n), 'constant', constant_values=0) if not isnull(arr) else arr.copy()
polyshift.__doc__ = "умножает мн-н на x^n"

def logger(f):
    def wrapper(*args, **kwargs):
        print(f"{f.__name__}({args}, {kwargs})")
        res = f(*args, **kwargs)
        print(f"-> {res}")
        return res
    return wrapper

def gen_pow_matrix(primpoly):
    """
    использование результата:
        alpha^i = pm[i - 1, 1]
    если b in GF, b != 0:
        j: (alpha^j = b) = pm[b - 1, 0] (j = j % mult_group_size)
    """
    q = _floorlog2(primpoly)
    pow2q = 2 ** q
    pow_vect = _np.empty(pow2q - 1, dtype=int)
    pow_vect[: q - 1] = _np.logspace(1, q - 1, num=q - 1, base=2, dtype=int)
    t = pow_vect[q - 2]
    for i in range(q - 1, pow2q - 2):
        t <<= 1
        if t & pow2q:
            t ^= primpoly
        pow_vect[i] = t
    pow_vect[pow2q - 2] = 1
    pm = _np.vstack((pow_vect.argsort() + 1, pow_vect)).T
    pm[0, 0] = 0
    return pm

get_pow = lambda p, pm: pm[p - 1, 0] if p != 0 else _np.nan
get_pow.__doc__ = \
"""get alpha**p
p in [0, 2**q - 1]"""

get_dec = lambda d, pm: pm[d - 1, 1]
get_dec.__doc__ = \
"""get p in [0, 2**q - 2]: alpha**p = d
d in [1, 2**q - 1]"""

def add(X, Y):
    if _np.isscalar(X) and _np.isscalar(Y):
        return X ^ Y
    else:
        return _np.bitwise_xor(X, Y)

def sum(X, axis=0):
    return _np.bitwise_xor.reduce(X, axis=axis)

def prod_div(X, Y, sign, pm):
    mgroup_size = len(pm)
    def prod_div_el(x, y):
        if x == 0:
            return 0
        if y == 0:
            return 0 if sign > 0 else _np.nan
        return get_dec((get_pow(x, pm=pm) + sign * get_pow(y, pm=pm)) % mgroup_size, pm=pm)
    if _np.isscalar(X) and _np.isscalar(Y):
        return prod_div_el(X, Y)
    else:
        return _np.vectorize(prod_div_el)(X, Y)

prod = lambda X, Y, pm: prod_div(X, Y, 1, pm=pm)
divide = lambda X, Y, pm: prod_div(X, Y, -1, pm=pm)

def pow(X, power, pm):
    """
    возведение матрицы X из элементов из поля Галуа, определяемого м-цей pm,
    в степень power
    """
    mgroup_size = len(pm)
    def pow_el(x):
        if x == 0:
            return 0
        else:
            return get_dec((get_pow(x, pm=pm) * power) % mgroup_size, pm=pm)
    if _np.isscalar(X):
        return pow_el(X)
    else:
        return _np.vectorize(pow_el)(X)

sq = lambda X, pm: pow(X, 2, pm=pm)
sq.__doc__ = "возведение в квадрат м-цы X из элементов из поля Галуа, определяемого м-цей pm"

def polyadd(x, y):
    """
    сложение многочленов из F2[x] или GF(2**q)[x]
    """
    x_len, y_len = len(x), len(y)
    if x_len > y_len:
        res = _np.bitwise_xor(_ljust0(y, x_len), x)
    elif x_len < y_len:
        res = _np.bitwise_xor(_ljust0(x, y_len), y)
    else:
        res = _np.bitwise_xor(x, y)
    if (res != 0).any():
        res = res[res.nonzero()[0][0]:]
    else:
        res = res[-1:]
    return res

binpolyadd = polyadd

def binpolyprod(x, y):
    """
    умножение многочленов из F2[x]
    """
    if isnull(x) or isnull(y):
        return null_poly()
    from functools import reduce
    y_deg = polydeg(y)
    x_deg = polydeg(x)
    x_, y_ = (x, y) if x_deg > y_deg else (y, x)
    x_deg, y_deg = (x_deg, y_deg) if x_deg > y_deg else (y_deg, x_deg)
    # x_buf: x_buf[:-1 - d] = x_ * x**(y_deg - d), d = 0,1,...,y_deg (x - переменная в полиноме)
    x_buf = polyshift(x_, y_deg + 1)
    product = reduce(binpolyadd, (x_buf[:-1 - i] for i, c in enumerate(y_) if c))
    return product

def binpolydiv(x, y):
    """
    деление многочленов из F2[x]
    """
    y_deg = polydeg(y)
    if y_deg == 0:
        if isnull(y):
            return _np.nan, _np.nan
        else:
            return _np.copy(x), null_poly()
    r = _np.copy(x)
    r_deg = polydeg(r)
    if r_deg < y_deg:
        return null_poly(), r
    deg = r_deg - y_deg
    q = _np.zeros(deg + 1, dtype=int)
    for _ in range(deg + 1):
        r = binpolyadd(r, polyshift(y, deg))
        q[-1 - deg] = 1
        r_deg = polydeg(r)
        if r_deg < y_deg:
            break
        deg = r_deg - y_deg
    else:
        raise Exception('error, out of cycle')
    return q, r

def linsolve(A, b, pm):
    # b - м.б. как строка, так и столбец
    # решаем методом гаусса
    Ab = _np.hstack((_np.array(A), _np.array(b).reshape(-1, 1)))
    size = len(A)
    for i in range(size):
        # отсортируем строки части матрицы в порядке возрастания индекса первого нуля в строке
        Ab_tosort = Ab[i:, :]
        for row in Ab_tosort[:, : -1]:
            if (row == 0).all():
                return _np.nan
        keys = [row.nonzero()[0][0] for row in Ab_tosort[:, : -1]]
        idx = _np.argsort(keys)
        Ab[i :, :] = Ab_tosort[idx]
        diag_el = Ab[i, i]
        if diag_el != 0:
            Ab[i, i :] = divide(Ab[i, i :], diag_el, pm=pm)
            if i >= size - 1:
                break
            row_copies = _np.repeat(Ab[i, i :].reshape(1, -1), size - i - 1, axis=0)
            factors = _np.repeat(Ab[i + 1 :, i].reshape(-1, 1), size + 1 - i, axis=1)
            Ab[i + 1 :, i :] = add(Ab[i + 1 :, i :], prod(row_copies, factors, pm=pm))
        else:
            return _np.nan
    # привели A к треугольному виду с единицами на диагонали
    #   (и одновременно с этим преобразовали b)
    # теперь надо привести A к единичному виду, преобразовывая вместе с этим b,
    #   но будем преобразовывать только b
    b_res = Ab[:, -1]
    for i in range(size - 1, 0, -1):
        b_res[: i] = add(b_res[: i], prod(b_res[i], Ab[:i, i], pm=pm))
    return b_res

def get_primpoly(pm):
    """
    получить по матрице pm порождающий эту группу неприводимый многочлен
    """
    q = _floorlog2(len(pm) + 1)
    primpoly = add(pm[q - 2, 1] << 1, pm[q - 1, 1])
    return binpoly_num2arr(primpoly)

def minpoly(x, pm):
    from itertools import combinations
    from functools import reduce, partial
    orbits = []
    orbit_x = None
    checked = set()
    found_x = False
    for root in x:
        found_x = False
        t = root
        if t in checked:
            continue
        orbit = {t}
        if t == 2:
            found_x = True
        while True:
            t = sq(t, pm=pm)
            if t == root:
                break
            if t == 2:
                found_x = True
            orbit.update({t})
        checked.update(orbit)
        if found_x:
            orbit_x = orbit
        else:
            orbits.append(orbit)
    if orbit_x is not None:
        res = get_primpoly(pm)
    else:
        res = [1]
    for orbit in orbits:
        # len(orbit) корней у м.м.
        # посчитаем коэфф-ты м.м. по ф-ле Виета,
        #   а многочлен с этими коэф-тами представим
        #   в виде двоичного числа
        deg = len(orbit)
        cur_minpoly = _np.ones(deg + 1, dtype=int)
        for k in range(1, deg + 1):
            coef = sum(list(map(
                lambda x: reduce(partial(prod, pm=pm), x),
                combinations(orbit, k)
            )))
            if coef not in {0,1}:
                raise Exception("programmer error")
            cur_minpoly[k] = coef
        res = binpolyprod(res, cur_minpoly)
    return res, _np.array(sorted(checked))

def polyval(p, x, pm):
    deg = polydeg(p)
    if deg == 0:
        return _np.full_like(x, p[0])
    powers = _np.ones((len(x), deg + 1), dtype=int)
    powers[:, 1] = x
    for d in range(2, deg + 1):
        powers[:, d] = pow(x, d, pm=pm)
    ans = sum(prod(_np.repeat(p[::-1].reshape(1, -1), len(x), axis=0), powers, pm=pm), axis=1)
    return ans

def polyprod(p1, p2, pm):
    p1_rev, p2_rev = (p1[::-1], p2[::-1]) if len(p1) > len(p2) else (p2[::-1], p1[::-1])
    deg1 = polydeg(p1_rev)
    deg2 = polydeg(p2_rev)
    deg = deg1 + deg2
    res_rev = _np.zeros(deg + 1, dtype=int)
    res_rev[0] = prod(p1_rev[0], p2_rev[0], pm=pm)
    for d in range(1, deg + 1):
        coef = 0
        for i in range(min(deg1, d), -1, -1):
            j = d - i
            if j > deg2:
                break
            coef ^= prod(p1_rev[i], p2_rev[j], pm=pm)
        res_rev[d] = coef
    return res_rev[::-1]

def polydiv(p1, p2, pm):
    deg2 = polydeg(p2)
    if deg2 == 0:
        if isnull(p2):
            return _np.nan, _np.nan
        else:
            return divide(p1, p2[0], pm=pm), null_poly()
    r = _np.copy(p1)
    r_deg = polydeg(r)
    if r_deg < deg2:
        return null_poly(), r
    # степень одночлена - части q, полученного на очередном шагу
    deg = r_deg - deg2
    q = _np.zeros(deg + 1, dtype=int)
    for _ in range(deg + 1):
        coef = divide(r[0], p2[0], pm=pm)
        q[-1 - deg] = coef
        # умножение p2 на одночлен coef * (x ** deg)
        # здесь можно не использовать polyprod
        # это умножение производится с минимальными затратами на вычисление
        product = polyshift(_np.concatenate((
            [r[0]], prod(p2[1:], coef, pm=pm)
        )), deg)
        r = polyadd(r, product)
        r_deg = polydeg(r)
        if r_deg < deg2:
            break
        deg = r_deg - deg2
    else:
        raise Exception('error, out of cycle')
    return q, r

def euclid(p1, p2, pm, max_deg=0):
    from collections import deque
    if polydeg(p1) >= polydeg(p2):
        xyr_buf = deque([[[1], null_poly(), p1], [null_poly(), [1], p2]])
    else:
        xyr_buf = deque([[null_poly(), [1], p2], [[1], null_poly(), p1]])
    x, y = xyr_buf[-1][:-1]
    while True:
        q, r = polydiv(xyr_buf[0][-1], xyr_buf[1][-1], pm=pm)
        x, y = (polyadd(xyr_buf[0][i], polyprod(xyr_buf[1][i], q, pm=pm)) for i in range(2))
        xyr_buf.append([x, y, r])
        xyr_buf.popleft()
        if polydeg(r) <= max_deg:
            break
    return r, x, y
