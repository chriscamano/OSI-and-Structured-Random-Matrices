// fwht_mex.c
#include "mex.h"
#include <math.h>

/* In-place Fast Walsh–Hadamard on x[0..m-1], m must be a power of two */
static void fwht(double *x, mwSize m) {
    for (mwSize len = 1; len < m; len <<= 1) {
        for (mwSize i = 0; i < m; i += len<<1) {
            for (mwSize j = 0; j < len; ++j) {
                double a = x[i + j];
                double b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1 || nlhs > 1)
        mexErrMsgIdAndTxt("fwht_mex:usage",
          "Y = fwht_mex(X): X must be real double, size m×n with m a power of two.");
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("fwht_mex:type",
          "Input must be a real double matrix.");

    mwSize m = mxGetM(prhs[0]), n = mxGetN(prhs[0]);
    if ((m & (m-1)) != 0)
        mexErrMsgIdAndTxt("fwht_mex:dim",
          "Number of rows m must be a power of two.");

    /* duplicate input into output */
    plhs[0] = mxDuplicateArray(prhs[0]);
    double *Y = mxGetPr(plhs[0]);
    double scale = 1.0/sqrt((double)m);

    /* apply FWHT + normalization column-wise */
    for (mwSize col = 0; col < n; ++col) {
        double *colPtr = Y + col*m;
        fwht(colPtr, m);
        for (mwSize i = 0; i < m; ++i)
            colPtr[i] *= scale;
    }
}
