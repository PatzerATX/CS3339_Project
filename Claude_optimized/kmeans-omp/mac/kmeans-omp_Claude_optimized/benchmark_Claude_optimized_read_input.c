#define _CRT_SECURE_NO_DEPRECATE 1

/* Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <chrono>
#include "benchmark_Claude_optimized_kmeans.h"
#include <unistd.h>
#include <iostream>

extern double wtime(void);

void usage(char *argv0) {
    const char *help =
        "\nUsage: %s [switches] -i filename\n\n"
        "    -i filename      :file containing data to be clustered\n"
        "    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
        "    -t threshold     :threshold value                       [default=0.001]\n"
        "    -l nloops        :iteration for each number of clusters [default=1]\n"
        "    -b               :input file is in binary format\n"
        "    -r               :calculate RMSE                        [default=off]\n"
        "    -o               :output cluster center coordinates     [default=off]\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

int setup(int argc, char **argv) {
    int  opt;
    extern char *optarg;
    char   *filename  = 0;
    float  *buf;
    char    line[1024];
    int     isBinaryFile = 0;

    float threshold    = 0.001f;
    int   max_nclusters = 5;
    int   min_nclusters = 5;
    int   best_nclusters = 0;
    int   nfeatures    = 0;
    int   npoints      = 0;
    float len;

    float **features;
    float **cluster_centres = NULL;
    int     i, j, index;
    int     nloops  = 1;
    int     isRMSE  = 0;
    float   rmse;
    int     isOutput = 0;

    while ((opt = getopt(argc, argv, "i:t:m:n:l:bro")) != EOF) {
        switch (opt) {
            case 'i': filename       = optarg;       break;
            case 'b': isBinaryFile   = 1;            break;
            case 't': threshold      = atof(optarg); break;
            case 'm': max_nclusters  = atoi(optarg); break;
            case 'n': min_nclusters  = atoi(optarg); break;
            case 'r': isRMSE         = 1;            break;
            case 'o': isOutput       = 1;            break;
            case 'l': nloops         = atoi(optarg); break;
            case '?': usage(argv[0]);                break;
            default:  usage(argv[0]);                break;
        }
    }

    if (filename == 0) usage(argv[0]);

    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &npoints,   sizeof(int));
        read(infile, &nfeatures, sizeof(int));

        /* Single contiguous allocation for the feature data */
        const int total = npoints * nfeatures;
        buf         = (float *) malloc(total * sizeof(float));
        features    = (float **)malloc(npoints * sizeof(float *));
        features[0] = (float *) malloc(total   * sizeof(float));
        for (i = 1; i < npoints; i++)
            features[i] = features[i - 1] + nfeatures;

        read(infile, buf, total * sizeof(float));
        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        /* Count lines (points) */
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                npoints++;
        rewind(infile);
        /* Count features from first data line */
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
                break;
            }
        }

        const int total = npoints * nfeatures;
        buf         = (float *) malloc(total * sizeof(float));
        features    = (float **)malloc(npoints * sizeof(float *));
        features[0] = (float *) malloc(total   * sizeof(float));
        for (i = 1; i < npoints; i++)
            features[i] = features[i - 1] + nfeatures;

        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j = 0; j < nfeatures; j++) {
                buf[i++] = atof(strtok(NULL, " ,\t\n"));
            }
        }
        fclose(infile);
    }

    printf("\nI/O completed\n");
    printf("\nNumber of objects: %d\n", npoints);
    printf("Number of features: %d\n", nfeatures);

    if (npoints < min_nclusters) {
        printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n",
               min_nclusters, npoints);
        exit(0);
    }

    srand(7);
    /* Single memcpy instead of element-wise copy */
    memcpy(features[0], buf, npoints * nfeatures * sizeof(float));
    free(buf);

    auto start = std::chrono::steady_clock::now();
    cluster_centres = NULL;
    index = cluster(npoints, nfeatures, features,
                    min_nclusters, max_nclusters,
                    threshold, &best_nclusters,
                    &cluster_centres, &rmse,
                    isRMSE, nloops);
    auto end  = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Kmeans core timing: " << time << " ms" << std::endl;

    if ((min_nclusters == max_nclusters) && (isOutput == 1)) {
        printf("\n================= Centroid Coordinates =================\n");
        for (i = 0; i < max_nclusters; i++) {
            printf("%d:", i);
            for (j = 0; j < nfeatures; j++)
                printf(" %.2f", cluster_centres[i][j]);
            printf("\n\n");
        }
    }

    len = (float)((max_nclusters - min_nclusters + 1) * nloops);

    printf("Number of Iteration: %d\n", nloops);

    if (min_nclusters != max_nclusters) {
        printf("Best number of clusters is %d\n", best_nclusters);
    } else {
        if (nloops != 1) {
            if (isRMSE)
                printf("Number of trials to approach the best RMSE of %.3f is %d\n",
                       rmse, index + 1);
        } else {
            if (isRMSE)
                printf("Root Mean Squared Error: %.3f\n", rmse);
        }
    }

    free(features[0]);
    free(features);
    free(cluster_centres[0]);
    free(cluster_centres);
    return 0;
}
