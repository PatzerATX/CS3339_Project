#include "getopt.h"

/*
 * The platform C library already provides getopt and the optarg/optind globals.
 * This compatibility unit preserves the original file structure without
 * compiling the legacy K&R GNU getopt implementation as C++.
 */
