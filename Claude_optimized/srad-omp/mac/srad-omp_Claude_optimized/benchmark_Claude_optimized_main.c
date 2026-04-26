#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "./benchmark_Claude_optimized_main.h"
#include "./util/graphics/graphics.h"
#include "./util/graphics/resize.h"
#include "./util/timer/timer.h"

int main(int argc, char* argv[]) {

  /* ── Timing variables ───────────────────────────────────────────────── */
  long long time0, time1, time2, time3, time4, time5;
  long long time6, time7, time8, time9, time10, time11, time12;

  time0 = get_time();

  /* ── Image / parameter declarations ────────────────────────────────── */
  fp* image_ori;
  int image_ori_rows, image_ori_cols;
  long image_ori_elem;

  fp* image;
  int Nr, Nc;
  long Ne;

  int niter;
  fp  lambda;

  int r1, r2, c1, c2;
  long NeROI;

  int *iN, *iS, *jE, *jW;

  int iter;
  long i, j;

  int mem_size_i, mem_size_j;
  int blocks_x, blocks_work_size, blocks_work_size2;
  size_t local_work_size;
  int no, mul;

  fp meanROI, meanROI2, varROI, q0sqr;

  time1 = get_time();

  /* ── Command-line arguments ─────────────────────────────────────────── */
  if (argc != 5) {
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  niter  = atoi(argv[1]);
  lambda = atof(argv[2]);
  Nr     = atoi(argv[3]);
  Nc     = atoi(argv[4]);

  time2 = get_time();

  /* ── Read image ─────────────────────────────────────────────────────── */
  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori      = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char* input_image_path = "../data/srad/image.pgm";
  if (!read_graphics(input_image_path, image_ori, image_ori_rows, image_ori_cols, 1)) {
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori != NULL) free(image_ori);
    return -1;
  }

  time3 = get_time();

  /* ── Resize ─────────────────────────────────────────────────────────── */
  Ne    = (long)Nr * Nc;   /* OPT-1: cast once, reuse; avoids repeated int→long promotion */
  image = (fp*)malloc(sizeof(fp) * Ne);
  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  time4 = get_time();

  /* ── Setup ROI and neighbour index tables ───────────────────────────── */
  r1 = 0;  r2 = Nr - 1;
  c1 = 0;  c2 = Nc - 1;
  NeROI = (long)(r2 - r1 + 1) * (c2 - c1 + 1);

  mem_size_i = sizeof(int) * Nr;
  mem_size_j = sizeof(int) * Nc;
  iN = (int*)malloc(mem_size_i);
  iS = (int*)malloc(mem_size_i);
  jW = (int*)malloc(mem_size_j);
  jE = (int*)malloc(mem_size_j);

  /* OPT-2: fill iN/iS with a single linear pass then fix two boundary
     entries, same for jW/jE.  Avoids separate boundary-condition loops. */
  for (i = 0; i < Nr; i++) { iN[i] = (int)i - 1; iS[i] = (int)i + 1; }
  for (j = 0; j < Nc; j++) { jW[j] = (int)j - 1; jE[j] = (int)j + 1; }
  iN[0]    = 0;      iS[Nr-1] = Nr - 1;
  jW[0]    = 0;      jE[Nc-1] = Nc - 1;

  /* ── Allocate working arrays ─────────────────────────────────────────── */
  fp *dN    = (fp*)malloc(sizeof(fp) * Ne);
  fp *dS    = (fp*)malloc(sizeof(fp) * Ne);
  fp *dW    = (fp*)malloc(sizeof(fp) * Ne);
  fp *dE    = (fp*)malloc(sizeof(fp) * Ne);
  fp *c     = (fp*)malloc(sizeof(fp) * Ne);
  fp *sums  = (fp*)malloc(sizeof(fp) * Ne);
  fp *sums2 = (fp*)malloc(sizeof(fp) * Ne);

  local_work_size  = NUMBER_THREADS;
  blocks_x         = (int)((Ne + local_work_size - 1) / local_work_size); /* OPT-3 */
  blocks_work_size = blocks_x;

  /* OPT-3: Replace two-step division+remainder with a single
     ceiling-division expression:  (Ne + T - 1) / T.
     At -O0 the original used an if-branch; the new form is branchless. */

  /* OPT-4: Pre-compute fp_NeROI reciprocal so the per-iteration
     meanROI / varROI divisions become multiplications.               */
  const fp inv_NeROI = FP_ONE / (fp)NeROI;

  time5 = get_time();

  /* ── OpenMP offload region ──────────────────────────────────────────── */
#pragma omp target data \
    map(to:   image[0:Ne]) \
    map(to:   iN[0:Nr], iS[0:Nr], jE[0:Nc], jW[0:Nc]) \
    map(alloc: dN[0:Ne], dS[0:Ne], dW[0:Ne], dE[0:Ne], \
               c[0:Ne], sums[0:Ne], sums2[0:Ne])
  {
    time6 = get_time();

    /* ── Expand: scale pixel values by expf(x/255) ─────────────────── */
    /* OPT-5: Pre-compute the reciprocal scale factor once outside the
       kernel.  At -O0 the compiler re-evaluates (fp)255 and the cast
       on every loop iteration.  Using a compile-time constant avoids
       both the cast and the division inside the kernel.               */
    const fp inv255 = FP_ONE / FP_255;

#pragma omp target teams distribute parallel for \
    num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++)
      image[ei] = expf(image[ei] * inv255);   /* multiply cheaper than divide */

    time7 = get_time();

    /* ── Main iteration loop ────────────────────────────────────────── */
    for (iter = 0; iter < niter; iter++) {

      /* ── Init sums / sums2 ─────────────────────────────────────── */
#pragma omp target teams distribute parallel for \
      num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
      for (int ei = 0; ei < Ne; ei++) {
        fp v      = image[ei];
        sums[ei]  = v;
        sums2[ei] = v * v;   /* OPT-6: cache image[ei] in local; avoids
                                 second global-memory load at -O0        */
      }

      /* ── Reduction ──────────────────────────────────────────────── */
      blocks_work_size2 = blocks_work_size;
      no  = (int)Ne;
      mul = 1;

      while (blocks_work_size2 != 0) {

#pragma omp target teams num_teams(blocks_work_size2) thread_limit(NUMBER_THREADS)
        {
          fp psum[NUMBER_THREADS];
          fp psum2[NUMBER_THREADS];
#pragma omp parallel
          {
            int bx  = omp_get_team_num();
            int tx  = omp_get_thread_num();
            int ei  = bx * NUMBER_THREADS + tx;
            int nf  = NUMBER_THREADS - (blocks_work_size2 * NUMBER_THREADS - no);
            int df  = 0;
            int i;

            if (ei < no) {
              /* OPT-7: compute index once into a local int to avoid
                 re-multiplication ei*mul at -O0 on every access      */
              int idx  = ei * mul;
              psum[tx]  = sums[idx];
              psum2[tx] = sums2[idx];
            }
#pragma omp barrier

            if (nf == NUMBER_THREADS) {
              for (i = 2; i <= NUMBER_THREADS; i <<= 1) { /* OPT-8: i*2 → i<<=1 */
                if ((tx + 1) % i == 0) {
                  psum[tx]  += psum[tx - (i >> 1)];       /* OPT-8: i/2 → i>>1  */
                  psum2[tx] += psum2[tx - (i >> 1)];
                }
#pragma omp barrier
              }
              if (tx == NUMBER_THREADS - 1) {
                int out = bx * mul * NUMBER_THREADS;
                sums[out]  = psum[tx];
                sums2[out] = psum2[tx];
              }
            } else {
              if (bx != blocks_work_size2 - 1) {
                for (i = 2; i <= NUMBER_THREADS; i <<= 1) {
                  if ((tx + 1) % i == 0) {
                    psum[tx]  += psum[tx - (i >> 1)];
                    psum2[tx] += psum2[tx - (i >> 1)];
                  }
#pragma omp barrier
                }
                if (tx == NUMBER_THREADS - 1) {
                  int out = bx * mul * NUMBER_THREADS;
                  sums[out]  = psum[tx];
                  sums2[out] = psum2[tx];
                }
              } else {
                /* last (possibly partial) block */
                for (i = 2; i <= NUMBER_THREADS; i <<= 1) {
                  if (nf >= i) df = i;
                }
                for (i = 2; i <= df; i <<= 1) {
                  if ((tx + 1) % i == 0 && tx < df) {
                    psum[tx]  += psum[tx - (i >> 1)];
                    psum2[tx] += psum2[tx - (i >> 1)];
                  }
#pragma omp barrier
                }
                if (tx == df - 1) {
                  int base = bx * NUMBER_THREADS;
                  int out  = bx * mul * NUMBER_THREADS;
                  for (i = base + df; i < base + nf; i++) {
                    psum[tx]  += sums[i];
                    psum2[tx] += sums2[i];
                  }
                  sums[out]  = psum[tx];
                  sums2[out] = psum2[tx];
                }
              }
            }
          } /* omp parallel */
        } /* omp target teams */

        no = blocks_work_size2;
        if (blocks_work_size2 == 1) {
          blocks_work_size2 = 0;
        } else {
          mul             *= (int)local_work_size; /* OPT-9: mul*NUMBER_THREADS → *= local_work_size */
          blocks_x         = (int)((blocks_work_size2 + local_work_size - 1) / local_work_size);
          blocks_work_size2 = blocks_x;
        }
      } /* while */

#pragma omp target update from(sums[0:1])
#pragma omp target update from(sums2[0:1])

      /* ── Statistics ─────────────────────────────────────────────── */
      /* OPT-4: multiply by pre-computed reciprocal instead of dividing */
      meanROI  = sums[0]  * inv_NeROI;
      meanROI2 = meanROI  * meanROI;
      varROI   = sums2[0] * inv_NeROI - meanROI2;
      q0sqr    = varROI   / meanROI2;

      /* OPT-10: pre-compute q0sqr-dependent constants once on CPU
         so the device kernel only fetches them, not recomputes them  */
      const fp q0sqr_p1    = FP_ONE + q0sqr;         /* 1 + q0sqr         */
      const fp q0sqr_denom = q0sqr * q0sqr_p1;       /* q0sqr*(1+q0sqr)   */

      /* ── Gradient / diffusion coefficient kernel ─────────────────── */
#pragma omp target teams distribute parallel for \
      num_teams(blocks_work_size) thread_limit(NUMBER_THREADS) \
      firstprivate(q0sqr, q0sqr_denom)
      for (int ei = 0; ei < Ne; ei++) {

        /* OPT-11: compute row/col with a single divmod pair.
           Original used two separate % and / per kernel (4 integer
           divides per element).  We compute them once.               */
        int tmp = ei + 1;
        int col = tmp / Nr;
        int row = tmp % Nr;
        if (row == 0) { row = Nr - 1; col -= 1; }
        else           { row -= 1; }

        fp d_Jc = image[ei];

        /* OPT-12: hoist repeated index arithmetic into local ints     */
        int iN_row = iN[row];
        int iS_row = iS[row];
        int jW_col = jW[col];
        int jE_col = jE[col];
        int Nr_col = Nr * col;

        fp N_loc = image[iN_row + Nr_col] - d_Jc;
        fp S_loc = image[iS_row + Nr_col] - d_Jc;
        fp W_loc = image[row + Nr * jW_col] - d_Jc;
        fp E_loc = image[row + Nr * jE_col] - d_Jc;

        /* OPT-13: strength-reduce repeated multiplications            */
        fp inv_Jc  = FP_ONE / d_Jc;
        fp inv_Jc2 = inv_Jc * inv_Jc;

        fp d_G2 = (N_loc*N_loc + S_loc*S_loc + W_loc*W_loc + E_loc*E_loc) * inv_Jc2;
        fp d_L  = (N_loc + S_loc + W_loc + E_loc) * inv_Jc;

        fp d_num  = FP_HALF * d_G2 - FP_INV16 * (d_L * d_L);
        fp d_den  = FP_ONE  + FP_QUARTER * d_L;
        fp d_qsqr = d_num  / (d_den * d_den);

        /* OPT-14: use pre-computed q0sqr_denom instead of
           recomputing q0sqr*(1+q0sqr) inside the kernel              */
        d_den       = (d_qsqr - q0sqr) / q0sqr_denom;
        fp d_c_loc  = FP_ONE / (FP_ONE + d_den);

        /* OPT-15: CLAMP01 macro → FCSEL on ARM64, no branch           */
        d_c_loc = CLAMP01(d_c_loc);

        dN[ei] = N_loc;
        dS[ei] = S_loc;
        dW[ei] = W_loc;
        dE[ei] = E_loc;
        c[ei]  = d_c_loc;
      }

      /* ── Image update kernel ─────────────────────────────────────── */
      /* OPT-16: pre-compute lambda*0.25 once; avoids fp multiply
         inside each thread at -O0                                     */
      const fp lam_quarter = FP_QUARTER * lambda;

#pragma omp target teams distribute parallel for \
      num_teams(blocks_work_size) thread_limit(NUMBER_THREADS) \
      firstprivate(lam_quarter)
      for (int ei = 0; ei < Ne; ei++) {

        /* OPT-11 applied here too: single divmod                      */
        int tmp = ei + 1;
        int col = tmp / Nr;
        int row = tmp % Nr;
        if (row == 0) { row = Nr - 1; col -= 1; }
        else           { row -= 1; }

        fp d_cN = c[ei];
        fp d_cS = c[iS[row] + Nr * col];
        fp d_cW = c[ei];
        fp d_cE = c[row + Nr * jE[col]];

        fp d_D = d_cN*dN[ei] + d_cS*dS[ei] + d_cW*dW[ei] + d_cE*dE[ei];

        /* OPT-16: multiply by pre-computed lam_quarter                */
        image[ei] += lam_quarter * d_D;
      }
    } /* iter loop */

    time8 = get_time();

    /* ── Compress: scale back to 0-255 ─────────────────────────────── */
#pragma omp target teams distribute parallel for \
    num_teams(blocks_work_size) thread_limit(NUMBER_THREADS)
    for (int ei = 0; ei < Ne; ei++)
      image[ei] = logf(image[ei]) * FP_255;  /* OPT-17: use named constant */

    time9 = get_time();

#pragma omp target update from(image[0:Ne])

    time10 = get_time();

    /* ── Write output ───────────────────────────────────────────────── */
    write_graphics("./image_out.pgm", image, Nr, Nc, 1, 255);

    time11 = get_time();

  } /* omp target data */

  /* ── Free memory ────────────────────────────────────────────────────── */
  free(image_ori);
  free(image);
  free(iN); free(iS); free(jW); free(jE);
  free(dN); free(dS); free(dW); free(dE);
  free(c);  free(sums); free(sums2);

  time12 = get_time();

  /* ── Timing report ──────────────────────────────────────────────────── */
  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
      (float)(time1-time0)/1000000, (float)(time1-time0)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
      (float)(time2-time1)/1000000, (float)(time2-time1)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
      (float)(time3-time2)/1000000, (float)(time3-time2)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n",
      (float)(time4-time3)/1000000, (float)(time4-time3)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
      (float)(time5-time4)/1000000, (float)(time5-time4)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
      (float)(time6-time5)/1000000, (float)(time6-time5)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n",
      (float)(time7-time6)/1000000, (float)(time7-time6)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n",
      (float)(time8-time7)/1000000, (float)(time8-time7)/(float)(time12-time0)*100, niter);
  printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n",
      (float)(time9-time8)/1000000, (float)(time9-time8)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n",
      (float)(time10-time9)/1000000, (float)(time10-time9)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n",
      (float)(time11-time10)/1000000, (float)(time11-time10)/(float)(time12-time0)*100);
  printf("%15.12f s, %15.12f %% : FREE MEMORY\n",
      (float)(time12-time11)/1000000, (float)(time12-time11)/(float)(time12-time0)*100);
  printf("Total time:\n%.12f s\n", (float)(time12-time0)/1000000);

  return 0;
}
