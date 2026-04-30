#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
#include <string.h>

#include "main_gemini_optimized.h"
#include "./util/graphics/graphics.h"
#include "./util/graphics/resize.h"
#include "./util/timer/timer.h"

int main(int argc, char* argv []) {
  long long time0, time12;
  time0 = get_time();

  fp* image_ori, *image;
  int image_ori_rows, image_ori_cols, Nr, Nc;
  long image_ori_elem, Ne, NeROI;
  int niter, r1, r2, c1, c2;
  fp lambda, meanROI, meanROI2, varROI, q0sqr;
  int* iN, *iS, *jE, *jW;

  if(argc != 5){
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  }
  niter = atoi(argv[1]);
  lambda = atof(argv[2]);
  Nr = atoi(argv[3]);
  Nc = atoi(argv[4]);
  Ne = Nr * Nc;

  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori = (fp*)aligned_alloc(128, sizeof(fp) * image_ori_elem);

  const char* input_image_path = "../data/srad/image.pgm";
  if (!read_graphics(input_image_path, image_ori, image_ori_rows, image_ori_cols, 1)) {
    printf("ERROR: failed to read input image\n");
    return -1;
  }

  image = (fp*)aligned_alloc(128, sizeof(fp) * Ne);
  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  r1 = 0; r2 = Nr - 1; c1 = 0; c2 = Nc - 1;
  NeROI = (r2 - r1 + 1) * (c2 - c1 + 1);

  iN = (int*)aligned_alloc(128, sizeof(int) * Nr);
  iS = (int*)aligned_alloc(128, sizeof(int) * Nr);
  jW = (int*)aligned_alloc(128, sizeof(int) * Nc);
  jE = (int*)aligned_alloc(128, sizeof(int) * Nc);

  for (int i = 0; i < Nr; i++) { iN[i] = i - 1; iS[i] = i + 1; }
  for (int j = 0; j < Nc; j++) { jW[j] = j - 1; jE[j] = j + 1; }
  iN[0] = 0; iS[Nr - 1] = Nr - 1; jW[0] = 0; jE[Nc - 1] = Nc - 1;

  fp *dN = (fp*)aligned_alloc(128, sizeof(fp) * Ne);
  fp *dS = (fp*)aligned_alloc(128, sizeof(fp) * Ne);
  fp *dW = (fp*)aligned_alloc(128, sizeof(fp) * Ne);
  fp *dE = (fp*)aligned_alloc(128, sizeof(fp) * Ne);
  fp *c = (fp*)aligned_alloc(128, sizeof(fp) * Ne);

  // Initial Extract
  #pragma omp parallel for
  for (int i = 0; i < Ne; i++) image[i] = expf(image[i] / 255.0f);

  for (int iter = 0; iter < niter; iter++) {
    double sum = 0, sum2 = 0;
    
    // Parallelized Reduction using OpenMP
    #pragma omp parallel reduction(+:sum, sum2)
    {
        #pragma omp for nowait
        for (int i = 0; i < Ne; i++) {
            sum += (double)image[i];
            sum2 += (double)(image[i] * image[i]);
        }
    }

    meanROI = (fp)(sum / NeROI);
    meanROI2 = meanROI * meanROI;
    varROI = (fp)(sum2 / NeROI) - meanROI2;
    q0sqr = varROI / meanROI2;

    // Gradient and Diffusion Coefficient calculation
    #pragma omp parallel for
    for (int col = 0; col < Nc; col++) {
      int col_base = col * Nr;
      int jW_idx = jW[col] * Nr;
      int jE_idx = jE[col] * Nr;
      
      for (int row = 0; row < Nr; row++) {
        int ei = col_base + row;
        fp d_Jc = image[ei];
        fp N_loc = image[iN[row] + col_base] - d_Jc;
        fp S_loc = image[iS[row] + col_base] - d_Jc;
        fp W_loc = image[row + jW_idx] - d_Jc;
        fp E_loc = image[row + jE_idx] - d_Jc;

        fp d_G2 = (N_loc*N_loc + S_loc*S_loc + W_loc*W_loc + E_loc*E_loc) / (d_Jc*d_Jc);
        fp d_L = (N_loc + S_loc + W_loc + E_loc) / d_Jc;
        fp d_num = (0.5f * d_G2) - (0.0625f * (d_L * d_L));
        fp d_den = 1.0f + (0.25f * d_L);
        fp d_qsqr = d_num / (d_den * d_den);
        
        fp d_c_loc = 1.0f / (1.0f + ((d_qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr))));
        if (d_c_loc < 0.0f) d_c_loc = 0.0f;
        else if (d_c_loc > 1.0f) d_c_loc = 1.0f;

        dN[ei] = N_loc; dS[ei] = S_loc; dW[ei] = W_loc; dE[ei] = E_loc;
        c[ei] = d_c_loc;
      }
    }

    // Divergence and Image Update
    #pragma omp parallel for
    for (int col = 0; col < Nc; col++) {
      int col_base = col * Nr;
      int jE_idx = jE[col] * Nr;
      for (int row = 0; row < Nr; row++) {
        int ei = col_base + row;
        fp d_cN = c[ei];
        fp d_cS = c[iS[row] + col_base];
        fp d_cW = c[ei];
        fp d_cE = c[row + jE_idx];
        fp d_D = d_cN * dN[ei] + d_cS * dS[ei] + d_cW * dW[ei] + d_cE * dE[ei];
        image[ei] += 0.25f * lambda * d_D;
      }
    }
  }

  // Compress
  #pragma omp parallel for
  for (int i = 0; i < Ne; i++) image[i] = logf(image[i]) * 255.0f;

  write_graphics("./image_out.pgm", image, Nr, Nc, 1, 255);

  free(image_ori); free(image); free(iN); free(iS); free(jW); free(jE);
  free(dN); free(dS); free(dW); free(dE); free(c);

  time12 = get_time();
  printf("Total time: %.12f s\n", (float)(time12 - time0) / 1000000);
  return 0;
}
