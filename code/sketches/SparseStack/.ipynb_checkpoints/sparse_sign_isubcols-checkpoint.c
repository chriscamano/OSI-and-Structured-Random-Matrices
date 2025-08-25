// #include <math.h> 
// #include "mex.h"
// 
// int log_base_d(long a, int d) {
//     int output = 0;
//     while (a > 0) {
//         a /= d;
//         output += 1;
//     }
//     return output-1;
// }
// 
// void mexFunction(int nlhs, mxArray *plhs[],
//                  int nrhs, const mxArray *prhs[]) {
// 
// 	mwSize d = (mwSize) mxGetScalar(prhs[0]);
// 	mwSize m = (mwSize) mxGetScalar(prhs[1]);
// 	mwSize zeta = (mwSize) mxGetScalar(prhs[2]);
// 	mwSize nnz = m*zeta;
// 
//     int subcolsize = (d-1)/zeta+1;
//     int subcolsize2 = d - (zeta-1)*subcolsize;
//     int idx_per_rand = log_base_d((long) RAND_MAX + 1, subcolsize);
//     int bit_per_rand = log_base_d((long) RAND_MAX + 1, 2);
// 
// 	if (zeta > d) zeta = d;
// 
// 	double lowval = -1/sqrt((double) zeta);
//     double increment = -2*lowval;
// 
//     plhs[0] = mxCreateSparse(d,m,nnz,false);
//     double *vals  = mxGetPr(plhs[0]);
//     mwIndex *rows = mxGetIr(plhs[0]);
//     mwIndex *colstarts = mxGetJc(plhs[0]);
// 
// 	// Set values
//     int myrand;
// 	for (int i = 0; i < nnz; i++) {
//         if (i % bit_per_rand == 0) myrand = rand();
//         vals[i] = (myrand % 2) * increment + lowval;;
//         myrand >>= 1;
// 	}
// 
// 	// Set column starts
// 	for (int i = 1; i < m+1; ++i){
// 	    colstarts[i] = i*zeta;
// 	}
// 
// 
//     for (int i = 0; i < nnz; ++i) {
//         if (i % idx_per_rand == 0) myrand = rand();
//         int j = i % zeta;
//         // use the smaller tail size when j == zeta-1
//         int size = (j == zeta - 1) ? subcolsize2 : subcolsize;
//         rows[i] = j * subcolsize + (myrand % size);
//         myrand /= size;
// }
// 
// }
// 

/* Compile in MATLAB with: mex -R2018a sparse_sign_isubcols.c */
#include <math.h>
#include <stdlib.h>
#include "mex.h"

static int ilog_base(long a, int d) {
    /* returns floor(log_d(a)) for a>=1, d>=2 */
    int out = 0;
    while (a > 0) { a /= d; out += 1; }
    return out - 1;
}

/* Unbiased uniform integer in [0, n) using rejection sampling */
static int uniform_int(int n) {
    /* n must be >= 1 */
    unsigned long limit = (unsigned long)RAND_MAX - ((unsigned long)RAND_MAX % (unsigned long)n);
    int r;
    do {
        r = rand();
    } while ((unsigned long)r > limit);
    return r % n;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {

    if (nrhs < 3) {
        mexErrMsgIdAndTxt("sparse_sign_isubcols:args",
                          "Usage: S = f(d, m, zeta)");
    }

    mwSize d    = (mwSize) mxGetScalar(prhs[0]);
    mwSize m    = (mwSize) mxGetScalar(prhs[1]);
    mwSize zeta = (mwSize) mxGetScalar(prhs[2]);

    if (d == 0 || m == 0 || zeta == 0) {
        plhs[0] = mxCreateSparse(d, m, 0, mxREAL);
        return;
    }

    if (zeta > d) zeta = d;

    /* guard against overflow in m*zeta */
    if (m > (mwSize)(~(mwSize)0 / zeta)) {
        mexErrMsgIdAndTxt("sparse_sign_isubcols:overflow",
                          "m*zeta overflows mwSize.");
    }
    mwSize nnz = m * zeta;

    /* block sizes: d = q*zeta + r with 0 <= r < zeta */
    mwSize q = d / zeta;
    mwSize r = d % zeta;

    plhs[0] = mxCreateSparse(d, m, nnz, mxREAL);
    double  *vals = mxGetPr(plhs[0]);
    mwIndex *rows = mxGetIr(plhs[0]);
    mwIndex *Jc   = mxGetJc(plhs[0]);

    /* column pointers */
    Jc[0] = 0;
    for (mwSize col = 0; col < m; ++col) {
        Jc[col + 1] = (mwIndex)((col + 1) * zeta);
    }

    /* sign magnitude and bit budget for rand() */
    const double a = 1.0 / sqrt((double) zeta);
    const int    bits_per_rand = ilog_base((long)RAND_MAX + 1L, 2);

    unsigned int sign_buf = 0U;
    int          bits_left = 0;

    mwSize p = 0;
    for (mwSize col = 0; col < m; ++col) {
        for (mwSize j = 0; j < zeta; ++j, ++p) {

            /* compute start and size for block j */
            mwSize size_j, start_j;
            if (j < r) {
                size_j = q + 1;
                start_j = j * (q + 1);
            } else {
                size_j = q;
                start_j = r * (q + 1) + (j - r) * q;
            }

            /* unbiased offset in [0, size_j) */
            int off = uniform_int((int)size_j);
            rows[p] = (mwIndex)(start_j + (mwSize)off);

            /* reuse bits of rand() for the sign */
            if (bits_left == 0) {
                sign_buf = (unsigned int)rand();
                bits_left = bits_per_rand;
            }
            int s = (sign_buf & 1U) ? 1 : -1;
            sign_buf >>= 1;
            --bits_left;

            vals[p] = s * a;
        }
    }
}
