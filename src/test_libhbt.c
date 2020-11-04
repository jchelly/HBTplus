#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "libHBT.h"


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  char config_file[500];
  int first_snap;
  int last_snap;
  if(comm_rank == 0) {
    if(argc != 4) {
      printf("Usage: test_libhbt config_file first_snap last_snap\n");
      exit(1);
    }
    strncpy(config_file, argv[1], 500);
    first_snap = atoi(argv[2]);
    last_snap = atoi(argv[3]); 
  }
  MPI_Bcast(config_file, 500, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&first_snap, 1,   MPI_INT,  0, MPI_COMM_WORLD);
  MPI_Bcast(&last_snap,  1,   MPI_INT,  0, MPI_COMM_WORLD);

  int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
  num_threads = omp_get_num_threads();
#endif

  hbt_init(config_file, num_threads);

  for(int snapnum=first_snap; snapnum<=last_snap; snapnum+=1)
    hbt_invoke(first_snap, snapnum, NULL, 0, NULL);

  hbt_free();

  MPI_Finalize();
  return 0;
}
