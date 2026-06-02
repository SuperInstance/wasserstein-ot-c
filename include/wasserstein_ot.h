#ifndef WASSERSTEIN_OT_H
#define WASSERSTEIN_OT_H

#include <stddef.h>

/* ========== Matrix type ========== */

typedef struct {
    double *data;
    size_t rows;
    size_t cols;
} wot_matrix_t;

/* Allocate a rows x cols matrix initialized to zero. Returns NULL on failure. */
wot_matrix_t *wot_mat_alloc(size_t rows, size_t cols);

/* Free a matrix and set the pointer to NULL. */
void wot_mat_free(wot_matrix_t *m);

/* Get element (i, j). */
static inline double wot_mat_get(const wot_matrix_t *m, size_t i, size_t j) {
    return m->data[i * m->cols + j];
}

/* Set element (i, j). */
static inline void wot_mat_set(wot_matrix_t *m, size_t i, size_t j, double val) {
    m->data[i * m->cols + j] = val;
}

/* Matrix multiply: C = A * B. Caller must ensure dimensions match. Returns NULL on failure. */
wot_matrix_t *wot_mat_multiply(const wot_matrix_t *a, const wot_matrix_t *b);

/* Transpose. Returns new matrix or NULL. */
wot_matrix_t *wot_mat_transpose(const wot_matrix_t *m);

/* Create an identity matrix of size n. */
wot_matrix_t *wot_mat_identity(size_t n);

/* Fill matrix with a constant. */
void wot_mat_fill(wot_matrix_t *m, double val);

/* Scale all elements by a scalar. */
void wot_mat_scale(wot_matrix_t *m, double s);

/* Check approximate equality element-wise (max abs diff < tol). */
int wot_mat_allclose(const wot_matrix_t *a, const wot_matrix_t *b, double tol);

/* ========== Sinkhorn algorithm (log-domain stabilized) ========== */

typedef struct {
    wot_matrix_t *transport_plan;  /* optimal coupling (rows x cols) */
    double distance;               /* regularized OT distance */
    int iterations;                /* number of iterations performed */
    double final_error;            /* final marginal error */
} wot_sinkhorn_result_t;

/* Free a sinkhorn result. */
void wot_sinkhorn_result_free(wot_sinkhorn_result_t *r);

/*
 * Run log-domain stabilized Sinkhorn algorithm.
 *
 * cost:    cost matrix (n x m)
 * a:       source histogram (n, must sum to 1)
 * b:       target histogram (m, must sum to 1)
 * reg:     entropic regularization parameter ( > 0)
 * max_iter: maximum iterations
 * tol:     convergence tolerance on marginal violations
 */
wot_sinkhorn_result_t *wot_sinkhorn(const wot_matrix_t *cost,
                                     const double *a, size_t a_len,
                                     const double *b, size_t b_len,
                                     double reg, int max_iter, double tol);

/* ========== Wasserstein distances ========== */

/*
 * Wasserstein-1 (Earth Mover's Distance) for discrete distributions on a line.
 * a, b: probability vectors (must sum to 1), length n.
 */
double wot_wasserstein_1(const double *a, const double *b, size_t n);

/*
 * Wasserstein-2 distance between discrete distributions.
 * Uses Sinkhorn with given regularization.
 * cost: n x m cost matrix. a (len n), b (len m) are probability vectors.
 */
double wot_wasserstein_2(const wot_matrix_t *cost,
                          const double *a, size_t a_len,
                          const double *b, size_t b_len,
                          double reg, int max_iter, double tol);

/* ========== Barycenter ========== */

/*
 * Compute the Wasserstein barycenter of K distributions.
 *
 * costs:     array of K cost matrices (each n x n).
 * measures:  array of K probability vectors (each of length n).
 * weights:   barycenter weights (length K, must sum to 1).
 * K:         number of measures.
 * n:         dimension of each measure.
 * reg:       regularization.
 * max_iter:  outer iterations for barycenter.
 * sinkhorn_max_iter: Sinkhorn iterations per inner solve.
 * tol:       convergence tolerance.
 *
 * Returns: probability vector of length n (caller must free).
 */
double *wot_barycenter(const wot_matrix_t **costs,
                        const double **measures,
                        const double *weights,
                        size_t K, size_t n,
                        double reg, int max_iter,
                        int sinkhorn_max_iter, double tol);

/* ========== JKO Gradient Flow ========== */

/*
 * One step of the Jordan-Kinderlehrer-Otto (JKO) gradient flow.
 *
 * cost:      cost matrix (n x n).
 * measure:   current distribution (length n).
 * grad:      gradient of the functional to minimize (length n).
 * step_size: JKO time step (tau > 0).
 * reg:       Sinkhorn regularization.
 * max_iter:  Sinkhorn iterations.
 * tol:       Sinkhorn tolerance.
 *
 * Returns: new distribution of length n (caller must free).
 */
double *wot_jko_step(const wot_matrix_t *cost,
                      const double *measure, size_t n,
                      const double *grad,
                      double step_size, double reg,
                      int max_iter, double tol);

/* ========== Utility ========== */

/* Normalize array to sum to 1 in-place. Returns actual sum before normalization. */
double wot_normalize(double *v, size_t n);

/* Check if array sums to ~1. */
int wot_is_probability(const double *v, size_t n, double tol);

/* Softmax in-place (numerically stable). */
void wot_softmax(double *v, size_t n);

#endif /* WASSERSTEIN_OT_H */
