#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "./main.h"
#include "./util/graphics/graphics.h"
#include "./util/graphics/resize.h"
#include "./util/timer/timer.h"

int main(int argc, char* argv[]) {

  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;
  long long time7;
  long long time8;
  long long time9;
  long long time10;
  long long time11;
  long long time12;

  time0 = get_time();

  fp* image_ori;
  int image_ori_rows;
  int image_ori_cols;
  long image_ori_elem;

  fp* image;
  int Nr, Nc;
  long Ne;

  int niter;
  fp lambda;

  int r1, r2, c1, c2;
  long NeROI;

  int* iN;
  int* iS;
  int* jE;
  int* jW;

  int iter;
  long i, j;

  int mem_size_i;
  int mem_size_j;

  int blocks_x;
  size_t local_work_size;
  fp meanROI;
  fp meanROI2;
  fp varROI;
  fp q0sqr;

  time1 = get_time();

  if (argc != 5) {
    printf("Usage: %s <repeat> <lambda> <number of rows> <number of columns>\n", argv[0]);
    return 1;
  } else {
    niter = atoi(argv[1]);
    lambda = atof(argv[2]);
    Nr = atoi(argv[3]);
    Nc = atoi(argv[4]);
  }

  time2 = get_time();

  image_ori_rows = 502;
  image_ori_cols = 458;
  image_ori_elem = image_ori_rows * image_ori_cols;
  image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

  const char* input_image_path = "./image.pgm";
  if (!read_graphics(input_image_path, image_ori, image_ori_rows, image_ori_cols, 1)) {
    printf("ERROR: failed to read input image at %s\n", input_image_path);
    if (image_ori != NULL) free(image_ori);
    return -1;
  }

  time3 = get_time();

  Ne = (long)Nr * (long)Nc;
  image = (fp*)malloc(sizeof(fp) * Ne);
  resize(image_ori, image_ori_rows, image_ori_cols, image, Nr, Nc, 1);

  time4 = get_time();

  r1 = 0;
  r2 = Nr - 1;
  c1 = 0;
  c2 = Nc - 1;
  NeROI = (long)(r2 - r1 + 1) * (long)(c2 - c1 + 1);

  mem_size_i = sizeof(int) * Nr;
  iN = (int*)malloc(mem_size_i);
  iS = (int*)malloc(mem_size_i);
  mem_size_j = sizeof(int) * Nc;
  jW = (int*)malloc(mem_size_j);
  jE = (int*)malloc(mem_size_j);

  for (i = 0; i < Nr; i++) {
    iN[i] = (int)i - 1;
    iS[i] = (int)i + 1;
  }
  for (j = 0; j < Nc; j++) {
    jW[j] = (int)j - 1;
    jE[j] = (int)j + 1;
  }

  iN[0] = 0;
  iS[Nr - 1] = Nr - 1;
  jW[0] = 0;
  jE[Nc - 1] = Nc - 1;

  fp* dN = (fp*)malloc(sizeof(fp) * Ne);
  fp* dS = (fp*)malloc(sizeof(fp) * Ne);
  fp* dW = (fp*)malloc(sizeof(fp) * Ne);
  fp* dE = (fp*)malloc(sizeof(fp) * Ne);
  fp* c = (fp*)malloc(sizeof(fp) * Ne);

  local_work_size = NUMBER_THREADS;
  blocks_x = (int)(Ne / (long)local_work_size);
  if (Ne % (long)local_work_size != 0) {
    blocks_x = blocks_x + 1;
  }
  (void)blocks_x;

  time5 = get_time();
  time6 = time5;

  for (long ei = 0; ei < Ne; ei++) {
    image[ei] = expf(image[ei] / (fp)255);
  }

  time7 = get_time();

  for (iter = 0; iter < niter; iter++) {
    fp sum = 0.0f;
    fp sum2 = 0.0f;

    for (long ei = 0; ei < Ne; ei++) {
      fp value = image[ei];
      sum += value;
      sum2 += value * value;
    }

    meanROI = sum / (fp)NeROI;
    meanROI2 = meanROI * meanROI;
    varROI = (sum2 / (fp)NeROI) - meanROI2;
    q0sqr = varROI / meanROI2;

    for (int col = 0; col < Nc; col++) {
      long col_offset = (long)Nr * col;
      long east_offset = (long)Nr * jE[col];
      long west_offset = (long)Nr * jW[col];

      for (int row = 0; row < Nr; row++) {
        long ei = col_offset + row;
        fp d_Jc = image[ei];

        fp N_loc = image[col_offset + iN[row]] - d_Jc;
        fp S_loc = image[col_offset + iS[row]] - d_Jc;
        fp W_loc = image[west_offset + row] - d_Jc;
        fp E_loc = image[east_offset + row] - d_Jc;

        fp d_G2 = (N_loc * N_loc + S_loc * S_loc + W_loc * W_loc + E_loc * E_loc) / (d_Jc * d_Jc);
        fp d_L = (N_loc + S_loc + W_loc + E_loc) / d_Jc;

        fp d_num = ((fp)0.5 * d_G2) - (((fp)1.0 / (fp)16.0) * (d_L * d_L));
        fp d_den = (fp)1.0 + ((fp)0.25 * d_L);
        fp d_qsqr = d_num / (d_den * d_den);

        d_den = (d_qsqr - q0sqr) / (q0sqr * ((fp)1.0 + q0sqr));
        fp d_c_loc = (fp)1.0 / ((fp)1.0 + d_den);

        if (d_c_loc < 0.0f) {
          d_c_loc = 0.0f;
        } else if (d_c_loc > 1.0f) {
          d_c_loc = 1.0f;
        }

        dN[ei] = N_loc;
        dS[ei] = S_loc;
        dW[ei] = W_loc;
        dE[ei] = E_loc;
        c[ei] = d_c_loc;
      }
    }

    for (int col = 0; col < Nc; col++) {
      long col_offset = (long)Nr * col;
      long east_offset = (long)Nr * jE[col];

      for (int row = 0; row < Nr; row++) {
        long ei = col_offset + row;
        fp d_cN = c[ei];
        fp d_cS = c[col_offset + iS[row]];
        fp d_cW = c[ei];
        fp d_cE = c[east_offset + row];

        fp d_D = d_cN * dN[ei] + d_cS * dS[ei] + d_cW * dW[ei] + d_cE * dE[ei];
        image[ei] += (fp)0.25 * lambda * d_D;
      }
    }
  }

  time8 = get_time();

  for (long ei = 0; ei < Ne; ei++) {
    image[ei] = logf(image[ei]) * (fp)255;
  }

  time9 = get_time();
  time10 = time9;

  write_graphics("./image_out.pgm", image, Nr, Nc, 1, 255);

  time11 = get_time();

  free(image_ori);
  free(image);
  free(iN);
  free(iS);
  free(jW);
  free(jE);
  free(dN);
  free(dS);
  free(dW);
  free(dE);
  free(c);

  time12 = get_time();

  printf("Time spent in different stages of the application:\n");
  printf("%15.12f s, %15.12f %% : SETUP VARIABLES\n",
      (float)(time1 - time0) / 1000000, (float)(time1 - time0) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : READ COMMAND LINE PARAMETERS\n",
      (float)(time2 - time1) / 1000000, (float)(time2 - time1) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : READ IMAGE FROM FILE\n",
      (float)(time3 - time2) / 1000000, (float)(time3 - time2) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : RESIZE IMAGE\n",
      (float)(time4 - time3) / 1000000, (float)(time4 - time3) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n",
      (float)(time5 - time4) / 1000000, (float)(time5 - time4) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO CPU->GPU\n",
      (float)(time6 - time5) / 1000000, (float)(time6 - time5) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : EXTRACT IMAGE\n",
      (float)(time7 - time6) / 1000000, (float)(time7 - time6) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : COMPUTE (%d iterations)\n",
      (float)(time8 - time7) / 1000000, (float)(time8 - time7) / (float)(time12 - time0) * 100, niter);
  printf("%15.12f s, %15.12f %% : COMPRESS IMAGE\n",
      (float)(time9 - time8) / 1000000, (float)(time9 - time8) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : COPY DATA TO GPU->CPU\n",
      (float)(time10 - time9) / 1000000, (float)(time10 - time9) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : SAVE IMAGE INTO FILE\n",
      (float)(time11 - time10) / 1000000, (float)(time11 - time10) / (float)(time12 - time0) * 100);
  printf("%15.12f s, %15.12f %% : FREE MEMORY\n",
      (float)(time12 - time11) / 1000000, (float)(time12 - time11) / (float)(time12 - time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", (float)(time12 - time0) / 1000000);

  return 0;
}
