# wasserstein-ot-c

Optimal transport computations in plain C11. No dependencies beyond `libm`.

## What's here

- **Sinkhorn algorithm** — log-domain stabilized, with convergence tracking
- **Wasserstein-1** — Earth Mover's Distance for discrete distributions on a line
- **Wasserstein-2** — via Sinkhorn on arbitrary cost matrices
- **Barycenter** — weighted Wasserstein barycenter of multiple distributions
- **JKO gradient flow** — single step of the Jordan-Kinderlehrer-Otto scheme

## Build

```sh
make
make test
```

## Usage

```c
#include "wasserstein_ot.h"

/* W1 distance */
double a[] = {1.0, 0.0, 0.0};
double b[] = {0.0, 0.0, 1.0};
double w1 = wot_wasserstein_1(a, b, 3);  // → 2.0

/* W2 distance */
wot_matrix_t *cost = /* your cost matrix */;
double w2 = wot_wasserstein_2(cost, a, 3, b, 3, 0.1, 1000, 1e-9);
```

## Design

All floating-point is `double`. Matrices are row-major. The Sinkhorn solver runs entirely in log-domain to avoid numerical issues with small regularization parameters.

## License

MIT
