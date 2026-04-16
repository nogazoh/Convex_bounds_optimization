# dc_fast.py

import numpy as np


class init_problem_from_model_fast:
    def __init__(self, y, D, h, p=3, C=10):
        self.y = y
        self.n = len(y)

        self.p = p  # number of domains
        self.C = C  # number of classes
        self.U = 1.0 / self.n  # unif dist
        self.eta = 0.01

        self.D = D
        self.h = h

        # Calculate the upper bound.
        self.load_M_fast()

        # compute H
        self.H = (1.0 / self.p) * h.sum(axis=1)

        self.verify_D_fast()

    def verify_D_fast(self):
        denom = np.sum(self.D, axis=tuple(range(self.D.ndim - 1)), keepdims=True)
        self.D[...] = self.D / denom

    # compatibility alias
    verify_D = verify_D_fast

    def load_M_fast(self):
        tmp = self.y[..., np.newaxis] - self.h
        tmp = tmp * tmp
        self.M = np.max(tmp)

    # compatibility alias
    load_M = load_M_fast

    def get_marginal_density(self):
        return self.D

    def get_regressor(self):
        return self.h

    def get_true_values(self):
        return self.y

    def get_H(self):
        return self.H


def euclidean_proj_simplex_fast(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex:
        min_w 0.5 * || w - v ||_2^2 , s.t. sum_i w_i = s, w_i >= 0
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    if v.sum() == s and np.all(v >= 0):
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w


# compatibility alias
euclidean_proj_simplex = euclidean_proj_simplex_fast


class ConvexConcaveSolverFast:
    def __init__(self, problem, seed, init_z="err", max_iter=100):
        self.problem = problem
        self.seed = seed
        self.max_iter = max_iter
        self.init_z = init_z

    def choose_init_z_fast(self, n=100):
        np.random.seed(self.seed)
        z_vals = np.random.rand(self.problem.p, n)
        o_vals = np.zeros(n)

        z_vals /= z_vals.sum(axis=0, keepdims=True)

        for i in range(n):
            if np.abs(z_vals[:, i].sum() - 1) > 1e-4:
                o_vals[i] = np.inf
            else:
                if self.init_z == "err":
                    tmp = self.problem.compute_err(z_vals[:, i])
                    if len(tmp.shape) == 2:
                        o_vals[i] = max(tmp.sum(axis=-1))
                    else:
                        o_vals[i] = max(tmp)
                elif self.init_z == "obj":
                    o_vals[i] = max(self.problem.compute_obj(z_vals[:, i]))

        k = np.argmin(o_vals)
        if np.abs(z_vals[:, k].sum() - 1) > 1e-8:
            print('somehow returning non norm z0', z_vals[:, k].sum() - 1, o_vals[k])
        return z_vals[:, k]

    # compatibility alias
    choose_init_z = choose_init_z_fast

    def print_iter(self, it, obj, err=None, sub_iter=False, disp=True):
        if disp:
            s = 'Iter {:d}: obj val={:0.4g}'.format(it, obj)
            if err is not None:
                s += '  err val={:0.4g}'.format(err)
            if sub_iter:
                s = '\t' + s
            print(s)

    def print_obj_increase(self, obj, delta, sub_iter=False, disp=True):
        if disp:
            s = 'Overshot obj ({:0.2g}): lowering delta ({:0.2g})'.format(obj, delta)
            if sub_iter:
                s = '\t' + s
            print(s)

    def check_converged(self, obj, obj_prev, sub_iter=False, disp=True):
        thresh = obj * 1e-8
        o_change = np.abs(obj - obj_prev)
        converged = False

        if o_change < thresh:
            s = 'Converged: change in values less than threshold ({:0.6g})'.format(o_change)
            converged = True
        elif obj < thresh:
            converged = True
            s = 'Converged: objective is less than threshold ({:0.4g})'.format(obj)

        if converged and disp:
            if sub_iter:
                s = '\t' + s
            print(s)

        return converged

    def solve_convex_iter_fast(self, zt, delta=1, max_iter=100, disp=False):
        g_obj = self.problem.compute_obj(zt)
        k = np.argmax(g_obj)

        vt = self.problem.compute_concave(zt)
        gvt = self.problem.compute_grad_concave(zt)
        z = zt.copy()
        last_change = 0

        o_iter = np.zeros(max_iter + 1)
        o_iter[0] = self.problem.compute_linearized_obj(zt, zt, vt, gvt)[k]
        self.print_iter(0, o_iter[0], sub_iter=True, disp=disp)

        for it in range(1, max_iter + 1):
            z_prev = z.copy()
            z = self.update_z_fast(k, z, gvt, delta)
            oi = self.problem.compute_linearized_obj(z_prev, z, vt, gvt)
            o_iter[it] = oi[k]

            change = True
            if o_iter[it] > o_iter[it - 1] or o_iter[it] < 0:
                z = z_prev.copy()
                delta = 0.1 * delta
                self.print_obj_increase(o_iter[it], delta, sub_iter=True, disp=disp)
                o_iter[it] = o_iter[it - 1]
                change = False
            elif self.check_converged(o_iter[it], o_iter[it - 1], sub_iter=True, disp=disp):
                break

            if change:
                last_change = it
            if it - last_change > 5:
                break

            self.print_iter(it, o_iter[it], sub_iter=True, disp=disp)

        return z

    # compatibility alias
    solve_convex_iter = solve_convex_iter_fast

    def update_z_fast(self, k, z, gvt, delta):
        z_grad = self.problem.linearized_obj_gradient(z, gvt)
        scale = 1.0
        z = z - delta * scale * np.array(z_grad[k, :]).flatten()
        return euclidean_proj_simplex_fast(z)

    # compatibility alias
    update_z = update_z_fast

    def project_onto_simplex_fast(self, z):
        z = z.copy()
        z[z < 0] = 0
        return z / z.sum()

    # compatibility alias
    project_onto_simplex = project_onto_simplex_fast

    def compute_err_obj_fast(self, z):
        err = self.problem.compute_sq_err(z)
        obj = max(self.problem.compute_obj(z))
        return err, obj

    # compatibility alias
    compute_err_obj = compute_err_obj_fast

    def solve_fast(self, z0=None, step=None, delta=1e-4):
        p = self.problem.p
        N = self.max_iter if step is None else step

        if z0 is None:
            z0 = self.choose_init_z_fast(n=100)

        if np.abs(z0.sum() - 1) > 1e-8:
            print('solve non norm z0', z0.sum() - 1)

        o_iter = np.zeros(N + 1)
        z_iter = np.zeros([p, N + 1])
        err_iter = np.zeros(N + 1)

        err_iter[0], o_iter[0] = self.compute_err_obj_fast(z0)
        z_iter[:, 0] = z0

        self.print_iter(0, o_iter[0], err=err_iter[0])

        DI = max(1, N // 10)
        for it in range(1, N + 1):
            z_iter[:, it] = self.solve_convex_iter_fast(z_iter[:, it - 1], delta=delta, disp=False)
            err_iter[it], o_iter[it] = self.compute_err_obj_fast(z_iter[:, it])
            self.print_iter(it, o_iter[it], err=err_iter[it], disp=(it % DI == 0))

            if o_iter[it - 1] < o_iter[it]:
                z_iter[:, it] = z_iter[:, it - 1].copy()
                err_iter[it], o_iter[it] = err_iter[it - 1], o_iter[it - 1]
                delta = 0.5 * delta
                print('\t\t lowering delta to {:0.4g}'.format(delta))
            elif self.check_converged(o_iter[it], o_iter[it - 1]):
                break

        print('Learned z', z_iter[:, it])
        print('Final Obj={:0.4g}'.format(o_iter[it]))
        print('')
        return z_iter[:, it], o_iter[it], err_iter[it]

    # compatibility alias
    solve = solve_fast


class ConvexConcaveProblemFast(object):
    def __init__(self, DP):
        self.D = DP.get_marginal_density()
        sc = self.D.sum(axis=1)
        self.D = self.D / sc.max()
        self.U = DP.U / sc.max()
        self.h = DP.get_regressor()
        self.y = DP.get_true_values()
        self.etaU = DP.eta * self.U
        self.M = DP.M
        self.p = DP.p
        self.H = 1.0 / DP.p * self.h.sum(axis=1)
        self.C = DP.C

    def compute_convex_fast(self, z):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        err = (hz - self.y) ** 2
        v0 = err - 2 * self.M * np.log(Kz)

        if self.D.ndim == 2:
            u = ((self.D + self.etaU) * v0[:, np.newaxis]).sum(axis=0)
        else:
            u = ((self.D + self.etaU) * v0[..., np.newaxis]).sum(axis=(0, 1))

        return u

    # compatibility alias
    compute_convex = compute_convex_fast

    def compute_concave_fast(self, z):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        err = (hz - self.y) ** 2

        EUlogKz = self.etaU * np.log(Kz).sum()
        LDzetaU = ((Dz + self.etaU) * err).sum()
        f = LDzetaU - 2 * EUlogKz

        if self.D.ndim == 2:
            EDlog = (self.D * np.log(Kz)[:, np.newaxis]).sum(axis=0)
        else:
            EDlog = (self.D * np.log(Kz)[..., np.newaxis]).sum(axis=(0, 1))

        v = -2 * self.M * EDlog + f
        return v

    # compatibility alias
    compute_concave = compute_concave_fast

    def compute_DzJzKzhz_fast(self, z):
        """
        Assumes D and h have domain index in last place.
        Either D = Dx in [N,p] or D = Dxy in [N,C,p] / [*,*,p]
        """
        if len(self.h.shape) == 2:
            n = len(self.h)

            z_mat = np.tile(z.flatten(), (self.D.shape[0], 1))
            zD = z_mat * self.D
            Dz_full = zD
            Jz_full = zD + self.etaU / self.p
            Kz_full = Dz_full + self.etaU
            hz_full = Jz_full / Kz_full

            hz_prob = np.zeros((n, self.C), dtype=hz_full.dtype)
            Jz_prob = np.zeros((n, self.C), dtype=Jz_full.dtype)

            rows = np.arange(n)
            h_int = self.h.astype(int)

            for k in range(self.p):
                hz_prob[rows, h_int[:, k]] += hz_full[:, k]
                Jz_prob[rows, h_int[:, k]] += Jz_full[:, k]

            hz = np.argmax(hz_prob, axis=1)
            Jz = np.argmax(Jz_prob, axis=1)
            Dz = Dz_full.sum(axis=-1)
            Kz = Dz + self.etaU

        else:
            Dh = self.D * self.h
            z_mat = np.tile(z.flatten(), (self.D.shape[0], self.D.shape[1], 1))
            zDh = z_mat * Dh
            Dz = (z_mat * self.D).sum(axis=-1)
            Jz = (zDh + self.etaU / self.p * self.h).sum(axis=-1)
            Kz = Dz + self.etaU
            hz = Jz / Kz

        return Dz, Jz, Kz, hz

    # compatibility alias
    compute_DzJzKzhz = compute_DzJzKzhz_fast

    def compute_grad_convex_fast(self, z):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        Dh = self.D * self.h

        gu = np.zeros([self.p, self.p])
        for k in range(self.p):
            a = 2 * (self.D[..., k] + self.etaU) / Kz
            for i in range(self.p):
                v0 = (hz - self.y) * Dh[..., i]
                v1 = ((hz - self.y) * hz + self.M) * self.D[..., i]
                gu[k, i] = (a * (v0 - v1)).sum()

        return np.matrix(gu)


    # compatibility alias
    compute_grad_convex = compute_grad_convex_fast

    def compute_grad_concave_fast(self, z):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        Dh = self.D * self.h

        gv = np.zeros([self.p, self.p])
        for k in range(self.p):
            a0 = hz - self.y
            a1 = 2 * self.M * (self.D[..., k] + self.etaU) / Kz
            a2 = a0 ** 2 - 2 * hz * a0 - a1
            for i in range(self.p):
                gv[k, i] = (a2 * self.D[..., i] + 2 * a0 * Dh[..., i]).sum()

        return np.matrix(gv)


    # compatibility alias
    compute_grad_concave = compute_grad_concave_fast

    def compute_sq_err_fast(self, z, ind=None):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        if ind is None:
            ind = np.arange(len(hz), dtype=int)
        return ((hz[ind] - self.y[ind]) ** 2).sum() / len(ind)

    # compatibility alias
    compute_sq_err = compute_sq_err_fast

    def compute_err_fast(self, z):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        err = (hz - self.y) ** 2
        return err

    # compatibility alias
    compute_err = compute_err_fast

    def compute_obj_fast(self, z):
        u = self.compute_convex_fast(z)
        v = self.compute_concave_fast(z)
        return u - v

    # compatibility alias
    compute_obj = compute_obj_fast

    def obj_gradient_fast(self, z):
        gu = self.compute_grad_convex_fast(z)
        gv = self.compute_grad_concave_fast(z)
        return gu - gv

    # compatibility alias
    obj_gradient = obj_gradient_fast

    def compute_linearized_obj_fast(self, z0, z, v0, gv0):
        u = self.compute_convex_fast(z)
        a0 = gv0 * (z - z0)[:, np.newaxis]
        return (u - v0)[:, np.newaxis] - np.array(a0)

    # compatibility alias
    compute_linearized_obj = compute_linearized_obj_fast

    def linearized_obj_gradient_fast(self, z, gv0):
        gu = self.compute_grad_convex_fast(z)
        return gu - gv0

    # compatibility alias
    linearized_obj_gradient = linearized_obj_gradient_fast


class ConvexConcaveProblemByClassFast(ConvexConcaveProblemFast):
    def __init__(self, D, h, y, eta=1e-20):
        self.D = D
        self.U = 1e-2 * D.mean()
        self.h = h
        self.y = y
        self.etaU = eta * self.U
        self.M = h.max() ** 2
        self.p = h.shape[-1]
        self.H = 1.0 / self.p * self.h.sum(axis=-1)

    def compute_sq_err_percls_fast(self, z, ind=None):
        Dz, Jz, Kz, hz = self.compute_DzJzKzhz_fast(z)
        if ind is None:
            ind = np.arange(hz.shape[1], dtype=int)
        return ((hz[:, ind] - self.y[:, ind]) ** 2).sum(axis=1) / len(ind)

    # compatibility alias
    compute_sq_err_percls = compute_sq_err_percls_fast

    def compute_sq_err_fast(self, z, ind=None):
        err_cls = self.compute_sq_err_percls_fast(z, ind=ind)
        return err_cls.sum()

    # compatibility alias
    compute_sq_err = compute_sq_err_fast