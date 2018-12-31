import gf
import numpy as _np

__all__ = ['BCH']

with open("primpoly.txt", "r") as f:
    primpoly_list = list(map(int, f.read().split(', ')))

def _lstrip0(arr):
    """удаляет нули слева в массиве (если в нём все нули, то оставляет один нуль)"""
    arr_ = _np.array(arr)
    return arr_[arr_.nonzero()[0][0] :] if (arr_ != 0).any() else gf.null_poly()

class BCH:
    def __init__(self, n, t, prim_choice=0, ref=None):
        """
        Построение БЧХ кода с параметрами n, t
        prim_choice - задаёт, который из возможных неприводимых многочленов
        (отсортированных по двоичной записи) выбрать для поля GF.
        В качестве ref можно задать уже построенный объект BCH
        с теми же n и prim_choice, но меньшим t, чтобы облегчить
        построение данного BCH (если эти условия не выполняются,
        ref игнорируется)
        """
        if '0' in bin(n)[2:]:
            raise ValueError("invalid n")
        self.n = n
        if 2 * t + 1 > n:
            raise ValueError("too big t")
        self.t = t
        self.prim_choice = prim_choice
        if ref is None or (ref.n, ref.prim_choice) != (self.n, self.prim_choice) or self.t < ref.t:
            primpolys = sorted(list(filter(lambda x: n + 1 < x <= n * 2 + 1, primpoly_list)))
            primpoly = primpolys[self.prim_choice]
            self.pm = gf.gen_pow_matrix(primpoly)
            self.R = self.pm[: 2 * self.t, 1]
            self.g, self.g_roots = gf.minpoly(self.R, self.pm)
        else:
            self.pm = ref.pm
            self.R = self.pm[: 2 * self.t, 1]
            if self.t > ref.t:
                new_roots = set(self.R) - set(ref.g_roots)
                if new_roots:
                    g_rem, new_g_roots = gf.minpoly(_np.array(list(new_roots)), pm=self.pm)
                    self.g = gf.binpolyprod(ref.g, g_rem)
                    self.g_roots = _np.array(sorted(list(set(ref.g_roots) | set(new_g_roots))))
                else:
                    self.g = ref.g
                    self.g_roots = ref.g_roots
            else:
                # self.t == ref.t
                self.g = ref.g
                self.g_roots = ref.g_roots
        # поделим x^n - 1 на g(x)
        binome = _np.pad(_np.zeros(n - 1, dtype=int), 1, mode="constant", constant_values=1)
        _, r = gf.binpolydiv(binome, self.g)
        if not gf.isnull(r):
            raise Exception("programmer error")
        self.m = gf.polydeg(self.g)
        self.k = self.n - self.m
        self.dist_ = None

    def encode(self, U):
        def encode_msg(u):
            u_poly = _lstrip0(u)
            q, r = gf.binpolydiv(gf.polyshift(u_poly, self.m), self.g)
            return _np.concatenate((u, gf._ljust0(r, self.m)))
        if _np.isscalar(U[0]):
            if len(U) != self.k:
                raise ValueError("message length must be {}".format(self.k))
            return encode_msg(U)
        else:
            if len(U[0]) != self.k:
                raise ValueError("message length must be {}".format(self.k))
            return _np.array(list(map(encode_msg, U)))

    def decode(self, W, method="euclid"):
        # м-ца степеней прим. эл-та данного поля Галуа, где индекс равен степени
        pm_powsfrom0 = _np.concatenate(([1], self.pm[: -1, 1].flatten()))

        def decode_msg(w):
            syndromes = gf.polyval(_lstrip0(w), self.R, pm=self.pm)
            if (syndromes == 0).all():
                return _np.copy(w)
            errloc_poly = decoder(w, syndromes)
            if errloc_poly is _np.nan:
                return _np.full(self.n, _np.nan)
            vals = gf.polyval(errloc_poly, pm_powsfrom0, pm=self.pm)
            roots_degs = _np.where(vals == 0)[0]
            v = _np.copy(w)
            v[roots_degs - 1] ^= 1
            _, r = gf.binpolydiv(_lstrip0(v), self.g)
            if not gf.isnull(r):
                return _np.full(self.n, _np.nan)
            v_syndromes = gf.polyval(_lstrip0(v), self.R, pm=self.pm)
            if (v_syndromes != 0).any():
                return _np.full(self.n, _np.nan)
            return v

        def decoder_pgz(w, syndromes):
            *A, b = (syndromes[i : i + self.t] for i in range(self.t + 1))
            A = _np.array(A)
            for n_err in range(self.t, 0, -1):
                sol = gf.linsolve(A, b, pm=self.pm)
                if sol is not _np.nan:
                    break
                b = A[-1, : -1]
                A = A[: -1, : -1]
            else:
                return _np.nan
            errloc_poly = _lstrip0(_np.concatenate((sol, [1])))
            return errloc_poly

        def decoder_euclid(w, syndromes):
            s_poly = _lstrip0(_np.concatenate((syndromes[::-1], [1])))
            _, _, errloc_poly = gf.euclid(
                gf.polyshift([1], 2 * self.t + 1),
                s_poly,
                pm=self.pm, max_deg=self.t + 1)
            return errloc_poly

        if method == "pgz":
            decoder = decoder_pgz
        elif method == "euclid":
            decoder = decoder_euclid
        else:
            raise ValueError("method not found")

        if _np.isscalar(W[0]):
            if len(W) != self.n:
                raise ValueError("message length must be {}".format(self.n))
            return decode_msg(W)
        else:
            if len(W[0]) != self.n:
                raise ValueError("message length must be {}".format(self.n))
            return _np.array(list(map(decode_msg, W)))

    def dist(self):
        if self.dist_ is not None:
            return self.dist_
        min_norm = self.n
        for num_repr in range(1, 2**self.k):
            u = list(map(int, _np.binary_repr(num_repr, width=self.k)))
            v = self.encode(u)
            norm = (v == 1).sum()
            if norm < min_norm:
                min_norm = norm
        self.dist_ = min_norm
        if self.dist_ < self.t * 2 + 1:
            raise Exception("programmer error")
        return self.dist_
