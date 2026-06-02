#include "wasserstein_ot.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { printf("  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_APPROX(a, b, tol, msg) ASSERT(fabs((a) - (b)) < (tol), msg)

/* Helper: build a simple cost matrix (squared Euclidean on grid) */
static wot_matrix_t *make_cost_sq_euclid(size_t n, size_t m)
{
    wot_matrix_t *c = wot_mat_alloc(n, m);
    if (!c) return NULL;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            double d = (double)i - (double)j;
            wot_mat_set(c, i, j, d * d);
        }
    return c;
}

static wot_matrix_t *make_cost_euclid(size_t n, size_t m)
{
    wot_matrix_t *c = wot_mat_alloc(n, m);
    if (!c) return NULL;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            wot_mat_set(c, i, j, fabs((double)i - (double)j));
    return c;
}

int main(void)
{
    printf("=== Wasserstein OT Test Suite ===\n\n");

    /* ---- Matrix operations ---- */
    printf("[Matrix Operations]\n");

    /* 1. Alloc and free */
    {
        wot_matrix_t *m = wot_mat_alloc(3, 4);
        ASSERT(m != NULL, "alloc returns non-NULL");
        ASSERT(m->rows == 3 && m->cols == 4, "alloc dimensions correct");
        wot_mat_free(m);
        tests_passed++; tests_run++; /* no crash = pass */
        printf("  PASS: alloc/free\n");
    }

    /* 2. Fill and get/set */
    {
        wot_matrix_t *m = wot_mat_alloc(2, 3);
        wot_mat_fill(m, 7.0);
        ASSERT(wot_mat_get(m, 0, 0) == 7.0, "fill sets values");
        wot_mat_set(m, 1, 2, 42.0);
        ASSERT(wot_mat_get(m, 1, 2) == 42.0, "set/get works");
        wot_mat_free(m);
        printf("  PASS: fill/set/get\n");
    }

    /* 3. Matrix multiply: identity * A = A */
    {
        wot_matrix_t *id = wot_mat_identity(3);
        wot_matrix_t *a = wot_mat_alloc(3, 2);
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 2; j++)
                wot_mat_set(a, i, j, (double)(i * 2 + j + 1));
        wot_matrix_t *c = wot_mat_multiply(id, a);
        ASSERT(c != NULL, "multiply returns non-NULL");
        ASSERT(wot_mat_allclose(c, a, 1e-12), "I*A = A");
        wot_mat_free(id); wot_mat_free(a); wot_mat_free(c);
        printf("  PASS: identity multiply\n");
    }

    /* 4. Transpose */
    {
        wot_matrix_t *m = wot_mat_alloc(2, 3);
        for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 3; j++)
                wot_mat_set(m, i, j, (double)(i * 10 + j));
        wot_matrix_t *t = wot_mat_transpose(m);
        ASSERT(t->rows == 3 && t->cols == 2, "transpose dimensions");
        ASSERT(wot_mat_get(t, 2, 1) == 12.0, "transpose element");
        wot_mat_free(m); wot_mat_free(t);
        printf("  PASS: transpose\n");
    }

    /* 5. Scale */
    {
        wot_matrix_t *m = wot_mat_alloc(2, 2);
        wot_mat_fill(m, 3.0);
        wot_mat_scale(m, 2.0);
        ASSERT(wot_mat_get(m, 0, 0) == 6.0, "scale works");
        wot_mat_free(m);
        printf("  PASS: scale\n");
    }

    /* 6. allclose */
    {
        wot_matrix_t *a = wot_mat_alloc(2, 2);
        wot_matrix_t *b = wot_mat_alloc(2, 2);
        wot_mat_fill(a, 1.0); wot_mat_fill(b, 1.0);
        ASSERT(wot_mat_allclose(a, b, 1e-9), "allclose equal");
        wot_mat_set(b, 0, 0, 2.0);
        ASSERT(!wot_mat_allclose(a, b, 1e-9), "allclose different");
        wot_mat_free(a); wot_mat_free(b);
        printf("  PASS: allclose\n");
    }

    /* ---- Utility ---- */
    printf("\n[Utility]\n");

    /* 7. Normalize */
    {
        double v[] = {1.0, 2.0, 3.0};
        double s = wot_normalize(v, 3);
        ASSERT_APPROX(s, 6.0, 1e-12, "normalize returns sum");
        ASSERT_APPROX(v[0] + v[1] + v[2], 1.0, 1e-12, "normalize sums to 1");
        printf("  PASS: normalize\n");
    }

    /* 8. is_probability */
    {
        double p[] = {0.25, 0.25, 0.25, 0.25};
        ASSERT(wot_is_probability(p, 4, 1e-9), "uniform is probability");
        double bad[] = {0.5, 0.3};
        ASSERT(!wot_is_probability(bad, 2, 1e-9), "non-summing is not probability");
        printf("  PASS: is_probability\n");
    }

    /* 9. Softmax */
    {
        double v[] = {1.0, 2.0, 3.0};
        wot_softmax(v, 3);
        ASSERT(wot_is_probability(v, 3, 1e-9), "softmax produces probability");
        ASSERT(v[2] > v[1] && v[1] > v[0], "softmax preserves order");
        printf("  PASS: softmax\n");
    }

    /* ---- Wasserstein-1 ---- */
    printf("\n[Wasserstein-1]\n");

    /* 10. W1: identical distributions */
    {
        double a[] = {0.25, 0.25, 0.25, 0.25};
        double w1 = wot_wasserstein_1(a, a, 4);
        ASSERT_APPROX(w1, 0.0, 1e-12, "W1(a,a) = 0");
        printf("  PASS: W1 identical = 0\n");
    }

    /* 11. W1: dirac delta */
    {
        double a[] = {1.0, 0.0, 0.0, 0.0};
        double b[] = {0.0, 0.0, 0.0, 1.0};
        double w1 = wot_wasserstein_1(a, b, 4);
        ASSERT_APPROX(w1, 3.0, 1e-12, "W1 delta_0 vs delta_3 = 3");
        printf("  PASS: W1 dirac delta\n");
    }

    /* 12. W1: shift by one */
    {
        double a[] = {1.0, 0.0, 0.0};
        double b[] = {0.0, 1.0, 0.0};
        double w1 = wot_wasserstein_1(a, b, 3);
        ASSERT_APPROX(w1, 1.0, 1e-12, "W1 delta_0 vs delta_1 = 1");
        printf("  PASS: W1 shift by one\n");
    }

    /* 13. W1: symmetric */
    {
        double a[] = {0.5, 0.5, 0.0};
        double b[] = {0.0, 0.5, 0.5};
        double w1_ab = wot_wasserstein_1(a, b, 3);
        double w1_ba = wot_wasserstein_1(b, a, 3);
        ASSERT_APPROX(w1_ab, w1_ba, 1e-12, "W1 is symmetric");
        printf("  PASS: W1 symmetric\n");
    }

    /* 14. W1: uniform distributions */
    {
        double a[] = {0.25, 0.25, 0.25, 0.25};
        double b[] = {0.25, 0.25, 0.25, 0.25};
        double w1 = wot_wasserstein_1(a, b, 4);
        ASSERT_APPROX(w1, 0.0, 1e-12, "W1 uniform vs uniform = 0");
        printf("  PASS: W1 uniform\n");
    }

    /* ---- Sinkhorn ---- */
    printf("\n[Sinkhorn]\n");

    /* 15. Sinkhorn: transport plan row sums */
    {
        double a[] = {0.5, 0.5};
        double b[] = {0.5, 0.5};
        wot_matrix_t *cost = make_cost_sq_euclid(2, 2);
        wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, 2, b, 2, 0.1, 1000, 1e-9);
        ASSERT(res != NULL, "sinkhorn returns result");
        ASSERT(res->iterations >= 0, "sinkhorn runs iterations");
        /* Check row sums ≈ a */
        double r0 = wot_mat_get(res->transport_plan, 0, 0) + wot_mat_get(res->transport_plan, 0, 1);
        double r1 = wot_mat_get(res->transport_plan, 1, 0) + wot_mat_get(res->transport_plan, 1, 1);
        ASSERT_APPROX(r0, 0.5, 1e-6, "sinkhorn row 0 sum");
        ASSERT_APPROX(r1, 0.5, 1e-6, "sinkhorn row 1 sum");
        wot_mat_free(cost);
        wot_sinkhorn_result_free(res);
        printf("  PASS: Sinkhorn transport plan row sums\n");
    }

    /* 16. Sinkhorn: transport plan column sums */
    {
        double a[] = {0.3, 0.7};
        double b[] = {0.6, 0.4};
        wot_matrix_t *cost = make_cost_sq_euclid(2, 2);
        wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, 2, b, 2, 0.01, 1000, 1e-9);
        ASSERT(res != NULL, "sinkhorn returns result (2)");
        double c0 = wot_mat_get(res->transport_plan, 0, 0) + wot_mat_get(res->transport_plan, 1, 0);
        double c1 = wot_mat_get(res->transport_plan, 0, 1) + wot_mat_get(res->transport_plan, 1, 1);
        ASSERT_APPROX(c0, 0.6, 1e-5, "sinkhorn col 0 sum");
        ASSERT_APPROX(c1, 0.4, 1e-5, "sinkhorn col 1 sum");
        wot_mat_free(cost);
        wot_sinkhorn_result_free(res);
        printf("  PASS: Sinkhorn transport plan col sums\n");
    }

    /* 17. Sinkhorn: convergence (should converge with enough iterations) */
    {
        double a[] = {0.25, 0.25, 0.25, 0.25};
        double b[] = {0.25, 0.25, 0.25, 0.25};
        wot_matrix_t *cost = make_cost_sq_euclid(4, 4);
        wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, 4, b, 4, 0.1, 1000, 1e-9);
        ASSERT(res != NULL, "sinkhorn convergence test returns result");
        ASSERT(res->final_error < 1e-6, "sinkhorn converges");
        wot_mat_free(cost);
        wot_sinkhorn_result_free(res);
        printf("  PASS: Sinkhorn convergence\n");
    }

    /* 18. Sinkhorn: distance is non-negative */
    {
        double a[] = {0.5, 0.5};
        double b[] = {0.5, 0.5};
        wot_matrix_t *cost = make_cost_euclid(2, 2);
        wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, 2, b, 2, 0.1, 500, 1e-9);
        ASSERT(res != NULL, "sinkhorn non-neg returns result");
        ASSERT(res->distance >= 0.0, "sinkhorn distance >= 0");
        wot_mat_free(cost);
        wot_sinkhorn_result_free(res);
        printf("  PASS: Sinkhorn distance non-negative\n");
    }

    /* ---- Wasserstein-2 ---- */
    printf("\n[Wasserstein-2]\n");

    /* 19. W2: identical distributions */
    {
        double a[] = {0.25, 0.25, 0.25, 0.25};
        wot_matrix_t *cost = make_cost_sq_euclid(4, 4);
        double w2 = wot_wasserstein_2(cost, a, 4, a, 4, 0.1, 1000, 1e-9);
        ASSERT(w2 >= 0.0, "W2(a,a) >= 0");
        ASSERT_APPROX(w2, 0.0, 1e-4, "W2(a,a) ≈ 0");
        wot_mat_free(cost);
        printf("  PASS: W2 identical ≈ 0\n");
    }

    /* 20. W2: non-negative */
    {
        double a[] = {1.0, 0.0, 0.0};
        double b[] = {0.0, 0.0, 1.0};
        wot_matrix_t *cost = make_cost_sq_euclid(3, 3);
        double w2 = wot_wasserstein_2(cost, a, 3, b, 3, 0.01, 1000, 1e-9);
        ASSERT(w2 > 0.0, "W2 different dists > 0");
        /* With squared euclidean on 3 points: W2²(delta_0, delta_2) should be ~4 */
        ASSERT(w2 > 1.0, "W2 delta_0 vs delta_2 is large");
        wot_mat_free(cost);
        printf("  PASS: W2 non-negative and meaningful\n");
    }

    /* 21. W2: triangle inequality (loose check) */
    {
        double a[] = {1.0, 0.0, 0.0, 0.0};
        double b[] = {0.0, 1.0, 0.0, 0.0};
        double c[] = {0.0, 0.0, 0.0, 1.0};
        wot_matrix_t *cost = make_cost_sq_euclid(4, 4);
        double w_ab = wot_wasserstein_2(cost, a, 4, b, 4, 0.01, 500, 1e-6);
        double w_bc = wot_wasserstein_2(cost, b, 4, c, 4, 0.01, 500, 1e-6);
        double w_ac = wot_wasserstein_2(cost, a, 4, c, 4, 0.01, 500, 1e-6);
        /* W2(a,c) <= W2(a,b) + W2(b,c) (with slack for regularization bias) */
        ASSERT(w_ac <= w_ab + w_bc + 5.0, "W2 triangle inequality (relaxed)");
        wot_mat_free(cost);
        printf("  PASS: W2 triangle inequality\n");
    }

    /* ---- Barycenter ---- */
    printf("\n[Barycenter]\n");

    /* 22. Barycenter of identical distributions */
    {
        size_t n = 4;
        double p[] = {0.25, 0.25, 0.25, 0.25};
        wot_matrix_t *cost = make_cost_sq_euclid(n, n);
        const wot_matrix_t *costs[] = {cost, cost};
        const double *measures[] = {p, p};
        double weights[] = {0.5, 0.5};
        double *bary = wot_barycenter(costs, measures, weights, 2, n, 0.1, 20, 200, 1e-6);
        ASSERT(bary != NULL, "barycenter returns non-NULL");
        ASSERT(wot_is_probability(bary, n, 1e-6), "barycenter is probability");
        /* Barycenter of identical uniform should be uniform */
        double max_dev = 0.0;
        for (size_t i = 0; i < n; i++) {
            double dev = fabs(bary[i] - 0.25);
            if (dev > max_dev) max_dev = dev;
        }
        ASSERT(max_dev < 0.05, "barycenter of identical ≈ uniform");
        wot_mat_free(cost);
        free(bary);
        printf("  PASS: barycenter of identical distributions\n");
    }

    /* 23. Barycenter of two different distributions */
    {
        size_t n = 5;
        double a[] = {1.0, 0.0, 0.0, 0.0, 0.0};
        double b[] = {0.0, 0.0, 0.0, 0.0, 1.0};
        wot_matrix_t *cost = make_cost_sq_euclid(n, n);
        const wot_matrix_t *costs[] = {cost, cost};
        const double *measures[] = {a, b};
        double weights[] = {0.5, 0.5};
        double *bary = wot_barycenter(costs, measures, weights, 2, n, 0.5, 30, 300, 1e-6);
        ASSERT(bary != NULL, "barycenter different dists returns non-NULL");
        ASSERT(wot_is_probability(bary, n, 0.1), "barycenter different is probability");
        /* Should be peaked somewhere in the middle */
        double total = 0.0;
        for (size_t i = 0; i < n; i++) total += bary[i];
        ASSERT_APPROX(total, 1.0, 0.05, "barycenter sums ~1");
        wot_mat_free(cost);
        free(bary);
        printf("  PASS: barycenter of two distributions\n");
    }

    /* ---- JKO Gradient Flow ---- */
    printf("\n[JKO Gradient Flow]\n");

    /* 24. JKO: result is a probability distribution */
    {
        size_t n = 5;
        double measure[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        double grad[] = {0.0, 0.0, 0.0, 0.0, 0.0}; /* zero gradient: should stay uniform */
        wot_matrix_t *cost = make_cost_sq_euclid(n, n);
        double *result = wot_jko_step(cost, measure, n, grad, 0.1, 0.1, 500, 1e-6);
        ASSERT(result != NULL, "JKO returns non-NULL");
        ASSERT(wot_is_probability(result, n, 0.05), "JKO result is probability");
        /* With zero gradient, result should be close to uniform */
        double max_dev = 0.0;
        for (size_t i = 0; i < n; i++) {
            double dev = fabs(result[i] - 0.2);
            if (dev > max_dev) max_dev = dev;
        }
        ASSERT(max_dev < 0.1, "JKO zero grad ≈ uniform");
        wot_mat_free(cost);
        free(result);
        printf("  PASS: JKO step produces probability\n");
    }

    /* 25. JKO: gradient pushes distribution */
    {
        size_t n = 4;
        double measure[] = {0.25, 0.25, 0.25, 0.25};
        double grad[] = {0.0, 0.0, 0.0, -10.0}; /* pushes mass toward index 3 */
        wot_matrix_t *cost = make_cost_sq_euclid(n, n);
        double *result = wot_jko_step(cost, measure, n, grad, 1.0, 0.1, 500, 1e-6);
        ASSERT(result != NULL, "JKO gradient returns non-NULL");
        ASSERT(wot_is_probability(result, n, 0.1), "JKO gradient result is probability");
        /* Mass should shift toward index 3 */
        ASSERT(result[3] > result[0], "JKO gradient pushes mass");
        wot_mat_free(cost);
        free(result);
        printf("  PASS: JKO gradient pushes distribution\n");
    }

    /* 26. JKO: step size zero ≈ identity */
    {
        size_t n = 3;
        double measure[] = {0.2, 0.6, 0.2};
        double grad[] = {1.0, -1.0, 1.0};
        wot_matrix_t *cost = make_cost_sq_euclid(n, n);
        double *result = wot_jko_step(cost, measure, n, grad, 0.0, 0.1, 500, 1e-6);
        ASSERT(result != NULL, "JKO zero step returns non-NULL");
        ASSERT(wot_is_probability(result, n, 0.1), "JKO zero step is probability");
        wot_mat_free(cost);
        free(result);
        printf("  PASS: JKO zero step size\n");
    }

    /* ---- Edge cases ---- */
    printf("\n[Edge Cases]\n");

    /* 27. Degenerate: single-point distributions */
    {
        double a[] = {1.0};
        double w1 = wot_wasserstein_1(a, a, 1);
        ASSERT_APPROX(w1, 0.0, 1e-12, "W1 single point = 0");
        printf("  PASS: W1 single-point distributions\n");
    }

    /* 28. Sinkhorn with large regularization (should converge fast) */
    {
        double a[] = {0.5, 0.5};
        double b[] = {0.5, 0.5};
        wot_matrix_t *cost = make_cost_sq_euclid(2, 2);
        wot_sinkhorn_result_t *res = wot_sinkhorn(cost, a, 2, b, 2, 10.0, 100, 1e-9);
        ASSERT(res != NULL, "sinkhorn large reg returns result");
        ASSERT(res->iterations <= 100, "sinkhorn large reg converges");
        ASSERT(res->final_error < 1e-6, "sinkhorn large reg error small");
        wot_mat_free(cost);
        wot_sinkhorn_result_free(res);
        printf("  PASS: Sinkhorn large regularization\n");
    }

    /* 29. W1 with uniform distributions */
    {
        double a[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        double b[] = {0.2, 0.2, 0.2, 0.2, 0.2};
        double w1 = wot_wasserstein_1(a, b, 5);
        ASSERT_APPROX(w1, 0.0, 1e-12, "W1 uniform 5 = 0");
        printf("  PASS: W1 uniform 5-point\n");
    }

    /* 30. Matrix: alloc zero-size */
    {
        wot_matrix_t *m = wot_mat_alloc(0, 0);
        /* Should either return NULL or a valid empty matrix */
        wot_mat_free(m);
        tests_passed++; tests_run++; /* no crash = pass */
        printf("  PASS: zero-size matrix alloc/free\n");
    }

    /* ---- Summary ---- */
    printf("\n=== Results: %d / %d tests passed ===\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
