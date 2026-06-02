#include "wasserstein_ot.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ========== Helper: log-sum-exp for numerical stability ========== */

static double logsumexp(const double *v, size_t n)
{
    if (n == 0) return -INFINITY;
    double max_v = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] > max_v) max_v = v[i];
    }
    if (isinf(max_v) && max_v < 0) return -INFINITY;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += exp(v[i] - max_v);
    }
    return max_v + log(sum);
}

/* ========== Matrix operations ========== */

wot_matrix_t *wot_mat_alloc(size_t rows, size_t cols)
{
    wot_matrix_t *m = malloc(sizeof(wot_matrix_t));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));
    if (!m->data) { free(m); return NULL; }
    return m;
}

void wot_mat_free(wot_matrix_t *m)
{
    if (m) {
        free(m->data);
        free(m);
    }
}

wot_matrix_t *wot_mat_multiply(const wot_matrix_t *a, const wot_matrix_t *b)
{
    if (a->cols != b->rows) return NULL;
    wot_matrix_t *c = wot_mat_alloc(a->rows, b->cols);
    if (!c) return NULL;
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < a->cols; k++) {
                sum += wot_mat_get(a, i, k) * wot_mat_get(b, k, j);
            }
            wot_mat_set(c, i, j, sum);
        }
    }
    return c;
}

wot_matrix_t *wot_mat_transpose(const wot_matrix_t *m)
{
    wot_matrix_t *t = wot_mat_alloc(m->cols, m->rows);
    if (!t) return NULL;
    for (size_t i = 0; i < m->rows; i++)
        for (size_t j = 0; j < m->cols; j++)
            wot_mat_set(t, j, i, wot_mat_get(m, i, j));
    return t;
}

wot_matrix_t *wot_mat_identity(size_t n)
{
    wot_matrix_t *m = wot_mat_alloc(n, n);
    if (!m) return NULL;
    for (size_t i = 0; i < n; i++) wot_mat_set(m, i, i, 1.0);
    return m;
}

void wot_mat_fill(wot_matrix_t *m, double val)
{
    for (size_t i = 0; i < m->rows * m->cols; i++) m->data[i] = val;
}

void wot_mat_scale(wot_matrix_t *m, double s)
{
    for (size_t i = 0; i < m->rows * m->cols; i++) m->data[i] *= s;
}

int wot_mat_allclose(const wot_matrix_t *a, const wot_matrix_t *b, double tol)
{
    if (a->rows != b->rows || a->cols != b->cols) return 0;
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        if (fabs(a->data[i] - b->data[i]) > tol) return 0;
    }
    return 1;
}

/* ========== Sinkhorn algorithm (log-domain stabilized) ========== */

void wot_sinkhorn_result_free(wot_sinkhorn_result_t *r)
{
    if (r) {
        wot_mat_free(r->transport_plan);
        free(r);
    }
}

wot_sinkhorn_result_t *wot_sinkhorn(const wot_matrix_t *cost,
                                     const double *a, size_t a_len,
                                     const double *b, size_t b_len,
                                     double reg, int max_iter, double tol)
{
    size_t n = a_len;
    size_t m = b_len;

    if (!cost || cost->rows != n || cost->cols != m || reg <= 0.0)
        return NULL;

    /* Log-domain Sinkhorn: u_i, v_j are log scaling vectors.
     * P_ij = exp(u_i + v_j - C_ij / reg)
     * Row constraint: exp(u_i) * sum_j exp(v_j - C_ij/reg) = a_i
     * Col constraint: exp(v_j) * sum_i exp(u_i - C_ij/reg) = b_j
     *
     * Update: u_i = log(a_i) - logsumexp_j(v_j - C_ij/reg)
     *         v_j = log(b_j) - logsumexp_i(u_i - C_ij/reg)
     */
    double *u = malloc(n * sizeof(double));
    double *v = calloc(m, sizeof(double)); /* v initialized to 0 */
    double *log_a = malloc(n * sizeof(double));
    double *log_b = malloc(m * sizeof(double));
    double *tmp = malloc((n > m ? n : m) * sizeof(double));

    if (!u || !v || !log_a || !log_b || !tmp) goto fail;

    for (size_t i = 0; i < n; i++) log_a[i] = (a[i] > 0) ? log(a[i]) : -INFINITY;
    for (size_t j = 0; j < m; j++) log_b[j] = (b[j] > 0) ? log(b[j]) : -INFINITY;

    /* Initialize u from v=0 */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            tmp[j] = v[j] - wot_mat_get(cost, i, j) / reg;
        }
        u[i] = log_a[i] - logsumexp(tmp, m);
    }

    int iter = 0;
    double err = 0.0;

    for (iter = 0; iter < max_iter; iter++) {
        /* Update v */
        for (size_t j = 0; j < m; j++) {
            for (size_t i = 0; i < n; i++) {
                tmp[i] = u[i] - wot_mat_get(cost, i, j) / reg;
            }
            v[j] = log_b[j] - logsumexp(tmp, n);
        }

        /* Update u */
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                tmp[j] = v[j] - wot_mat_get(cost, i, j) / reg;
            }
            u[i] = log_a[i] - logsumexp(tmp, m);
        }

        /* Check convergence: column marginals */
        if (iter % 10 == 0) {
            err = 0.0;
            for (size_t j = 0; j < m; j++) {
                for (size_t i = 0; i < n; i++) {
                    tmp[i] = u[i] + v[j] - wot_mat_get(cost, i, j) / reg;
                }
                double col_sum = exp(logsumexp(tmp, n));
                err += fabs(col_sum - b[j]);
            }
            if (err < tol) break;
        }
    }

    /* Build transport plan: P_ij = exp(u_i + v_j - C_ij / reg) */
    wot_matrix_t *plan = wot_mat_alloc(n, m);
    if (!plan) goto fail;

    double dist = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            double val = exp(u[i] + v[j] - wot_mat_get(cost, i, j) / reg);
            wot_mat_set(plan, i, j, val);
            dist += val * wot_mat_get(cost, i, j);
        }
    }

    wot_sinkhorn_result_t *result = malloc(sizeof(wot_sinkhorn_result_t));
    if (!result) { wot_mat_free(plan); goto fail; }
    result->transport_plan = plan;
    result->distance = dist;
    result->iterations = iter;
    result->final_error = err;

    free(u); free(v); free(log_a); free(log_b); free(tmp);
    return result;

fail:
    free(u); free(v); free(log_a); free(log_b); free(tmp);
    return NULL;
}

/* ========== Wasserstein distances ========== */

double wot_wasserstein_1(const double *a, const double *b, size_t n)
{
    /* Earth Mover's Distance on a uniform grid of spacing 1 */
    double diff = 0.0;
    double emd = 0.0;
    for (size_t i = 0; i < n; i++) {
        diff += a[i] - b[i];
        emd += fabs(diff);
    }
    return emd;
}

double wot_wasserstein_2(const wot_matrix_t *cost,
                          const double *a, size_t a_len,
                          const double *b, size_t b_len,
                          double reg, int max_iter, double tol)
{
    wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, a_len, b, b_len, reg, max_iter, tol);
    if (!res) return -1.0;
    double dist = res->distance;
    wot_sinkhorn_result_free(res);
    return dist;
}

/* ========== Barycenter ========== */

double *wot_barycenter(const wot_matrix_t **costs,
                        const double **measures,
                        const double *weights,
                        size_t K, size_t n,
                        double reg, int max_iter,
                        int sinkhorn_max_iter, double tol)
{
    /* Initialize barycenter as uniform */
    double *bary = malloc(n * sizeof(double));
    if (!bary) return NULL;
    for (size_t i = 0; i < n; i++) bary[i] = 1.0 / (double)n;

    double *log_bary = malloc(n * sizeof(double));
    double *new_log = malloc(n * sizeof(double));
    if (!log_bary || !new_log) { free(bary); free(log_bary); free(new_log); return NULL; }

    for (int iter = 0; iter < max_iter; iter++) {
        for (size_t i = 0; i < n; i++) log_bary[i] = log(bary[i]);

        /* weighted average of log dual potentials (fixed-point iteration) */
        for (size_t i = 0; i < n; i++) new_log[i] = 0.0;

        for (size_t k = 0; k < K; k++) {
            wot_sinkhorn_result_t *res = wot_sinkhorn(costs[k], bary, n,
                                                       measures[k], n,
                                                       reg, sinkhorn_max_iter, tol);
            if (!res) { free(bary); free(log_bary); free(new_log); return NULL; }

            /* Accumulate weighted transport plan marginals */
            for (size_t i = 0; i < n; i++) {
                double col_sum = 0.0;
                for (size_t j = 0; j < n; j++) {
                    col_sum += wot_mat_get(res->transport_plan, j, i);
                }
                if (col_sum > 0) {
                    new_log[i] += weights[k] * log(col_sum);
                }
            }
            wot_sinkhorn_result_free(res);
        }

        /* Update barycenter */
        double total = 0.0;
        for (size_t i = 0; i < n; i++) {
            bary[i] = exp(new_log[i]);
            total += bary[i];
        }
        if (total > 0) {
            for (size_t i = 0; i < n; i++) bary[i] /= total;
        }
    }

    free(log_bary);
    free(new_log);
    return bary;
}

/* ========== JKO Gradient Flow ========== */

double *wot_jko_step(const wot_matrix_t *cost,
                      const double *measure, size_t n,
                      const double *grad,
                      double step_size, double reg,
                      int max_iter, double tol)
{
    /* Modified cost: C_ij = cost_ij + step_size * grad_j */
    wot_matrix_t *mod_cost = wot_mat_alloc(n, n);
    if (!mod_cost) return NULL;

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            wot_mat_set(mod_cost, i, j, wot_mat_get(cost, i, j) + step_size * grad[j]);

    /* Target is uniform — we transport from measure to find the proximal map */
    double *target = malloc(n * sizeof(double));
    if (!target) { wot_mat_free(mod_cost); return NULL; }
    for (size_t i = 0; i < n; i++) target[i] = 1.0 / (double)n;

    wot_sinkhorn_result_t *res = wot_sinkhorn(mod_cost, measure, n, target, n,
                                               reg, max_iter, tol);
    wot_mat_free(mod_cost);

    if (!res) { free(target); return NULL; }

    /* New distribution = column marginals of transport plan */
    double *new_measure = calloc(n, sizeof(double));
    if (!new_measure) { wot_sinkhorn_result_free(res); free(target); return NULL; }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            new_measure[j] += wot_mat_get(res->transport_plan, i, j);
        }
    }

    wot_normalize(new_measure, n);
    wot_sinkhorn_result_free(res);
    free(target);
    return new_measure;
}

/* ========== Utility ========== */

double wot_normalize(double *v, size_t n)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += v[i];
    if (sum > 0.0) {
        for (size_t i = 0; i < n; i++) v[i] /= sum;
    }
    return sum;
}

int wot_is_probability(const double *v, size_t n, double tol)
{
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (v[i] < -tol) return 0;
        sum += v[i];
    }
    return fabs(sum - 1.0) < tol;
}

void wot_softmax(double *v, size_t n)
{
    double max_v = v[0];
    for (size_t i = 1; i < n; i++) if (v[i] > max_v) max_v = v[i];
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) { v[i] = exp(v[i] - max_v); sum += v[i]; }
    if (sum > 0) for (size_t i = 0; i < n; i++) v[i] /= sum;
}
