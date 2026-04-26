/* benchmark_Claude_optimized_material.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

SimulationData get_materials(Input input, uint64_t * seed)
{
    SimulationData SD;
    SD.num_nucs = load_num_nucs(input);
    SD.length_num_nucs = 12;
    SD.mats = load_mats(input, SD.num_nucs, &SD.max_num_nucs, &SD.length_mats);
    SD.concs = load_concs(SD.num_nucs, seed, SD.max_num_nucs);
    SD.length_concs = 12 * SD.max_num_nucs;
    return SD;
}

int * load_num_nucs(Input input)
{
    int * num_nucs = (int*)malloc(12 * sizeof(int));

    if (input.n_nuclides == 68)
        num_nucs[0] = 34;
    else
        num_nucs[0] = 321;

    num_nucs[1]  = 5;
    num_nucs[2]  = 4;
    num_nucs[3]  = 4;
    num_nucs[4]  = 27;
    num_nucs[5]  = 21;
    num_nucs[6]  = 21;
    num_nucs[7]  = 21;
    num_nucs[8]  = 21;
    num_nucs[9]  = 21;
    num_nucs[10] = 9;
    num_nucs[11] = 9;

    return num_nucs;
}

int * load_mats(Input input, int * num_nucs, int * max_num_nucs,
                unsigned long * length_mats)
{
    *max_num_nucs = 0;
    const int num_mats = 12;
    for (int m = 0; m < num_mats; m++)
        if (num_nucs[m] > *max_num_nucs) *max_num_nucs = num_nucs[m];

    int * mats = (int *) malloc(num_mats * (*max_num_nucs) * sizeof(int));
    *length_mats = (unsigned long)(num_mats * (*max_num_nucs));

    static const int mats0_Sml[] = {
        58,59,60,61,40,42,43,44,45,46,1,2,3,7,
        8,9,10,29,57,47,48,0,62,15,33,34,52,53,
        54,55,56,18,23,41
    };
    int mats0_Lrg[321];
    memcpy(mats0_Lrg, mats0_Sml, 34 * sizeof(int));
    for (int i = 0; i < 321-34; i++) mats0_Lrg[34+i] = 68 + i;

    static const int mats1[]  = {63,64,65,66,67};
    static const int mats2[]  = {24,41,4,5};
    static const int mats3[]  = {24,41,4,5};
    static const int mats4[]  = {19,20,21,22,35,36,37,38,39,25,27,28,29,
                                  30,31,32,26,49,50,51,11,12,13,14,6,16,17};
    static const int mats5[]  = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                                  49,50,51,11,12,13,14};
    static const int mats6[]  = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                                  49,50,51,11,12,13,14};
    static const int mats7[]  = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                                  49,50,51,11,12,13,14};
    static const int mats8[]  = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                                  49,50,51,11,12,13,14};
    static const int mats9[]  = {24,41,4,5,19,20,21,22,35,36,37,38,39,25,
                                  49,50,51,11,12,13,14};
    static const int mats10[] = {24,41,4,5,63,64,65,66,67};
    static const int mats11[] = {24,41,4,5,63,64,65,66,67};

    /* OPT-28: Mark constant material arrays as 'static const' so they live
       in read-only data segment rather than being re-initialised on the
       stack on every call to load_mats at -O0.                             */
    const int mn = *max_num_nucs;
    if (input.n_nuclides == 68)
        memcpy(mats,           mats0_Sml, num_nucs[0]  * sizeof(int));
    else
        memcpy(mats,           mats0_Lrg, num_nucs[0]  * sizeof(int));

    memcpy(mats + mn*1,  mats1,  num_nucs[1]  * sizeof(int));
    memcpy(mats + mn*2,  mats2,  num_nucs[2]  * sizeof(int));
    memcpy(mats + mn*3,  mats3,  num_nucs[3]  * sizeof(int));
    memcpy(mats + mn*4,  mats4,  num_nucs[4]  * sizeof(int));
    memcpy(mats + mn*5,  mats5,  num_nucs[5]  * sizeof(int));
    memcpy(mats + mn*6,  mats6,  num_nucs[6]  * sizeof(int));
    memcpy(mats + mn*7,  mats7,  num_nucs[7]  * sizeof(int));
    memcpy(mats + mn*8,  mats8,  num_nucs[8]  * sizeof(int));
    memcpy(mats + mn*9,  mats9,  num_nucs[9]  * sizeof(int));
    memcpy(mats + mn*10, mats10, num_nucs[10] * sizeof(int));
    memcpy(mats + mn*11, mats11, num_nucs[11] * sizeof(int));

    return mats;
}

double * load_concs(int * num_nucs, uint64_t * seed, int max_num_nucs)
{
    double * concs = (double *) malloc(12 * max_num_nucs * sizeof(double));

    for (int i = 0; i < 12; i++)
    {
        /* OPT-25 applied: cache row base                                    */
        const int row = i * max_num_nucs;
        for (int j = 0; j < num_nucs[i]; j++)
            concs[row + j] = LCG_RAND_DOUBLE(*seed);
    }

    return concs;
}
