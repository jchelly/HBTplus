using namespace std;
#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>

#include "src/mpi_wrapper.h"
#include "src/datatypes.h"
#include "src/config_parser.h"
#include "src/snapshot.h"
#include "src/halo.h"
#include "src/subhalo.h"
#include "src/mymath.h"
#include "src/particle_exchanger.h"

#include "libHBT.h"


// Data which must persist between HBT invocations
static MpiWorker_t *world_ptr;
static int num_hbt_threads;
static SubhaloSnapshot_t *subsnap_ptr;


void hbt_init(char *config_file, int num_threads)
{
  // MPI setup
  world_ptr = new MpiWorker_t(MPI_COMM_WORLD);
  MpiWorker_t &world = (*world_ptr);

  // OpenMP configuration
  num_hbt_threads = num_threads;

  // Read input file and broadcast parameters
  if(0==world.rank())
    {
      HBTConfig.ParseConfigFile(config_file);
      mkdir(HBTConfig.SubhaloPath.c_str(), 0755);
      HBTConfig.DumpParameters();
    
      cout<<"libHBT called using "<<world.size()<<" mpi tasks";
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
      cout<<", each with "<<omp_get_num_threads()<<" threads";
#endif
      cout<<endl;
    }
  HBTConfig.BroadCast(world, 0);

  // Initially we have no subhalo data in memory
  subsnap_ptr = NULL;
}


void hbt_invoke(int first_snapnum, int this_snapnum)
{
  MpiWorker_t &world = (*world_ptr);  
#ifdef _OPENMP
  omp_set_nested(0);
  omp_set_num_threads(num_hbt_threads);
#endif

  // Read in subhalos from previous snapshot if necessary
  if(!subsnap_ptr)
    {
      subsnap_ptr = new SubhaloSnapshot_t;
      (*subsnap_ptr).Load(world, this_snapnum-1, SubReaderDepth_t::SrcParticles);
    }
  SubhaloSnapshot_t &subsnap = (*subsnap_ptr);

  // Load particles and halos for this output
  ParticleSnapshot_t partsnap;
  partsnap.Load(world, this_snapnum);
  subsnap.SetSnapshotIndex(this_snapnum);
  HaloSnapshot_t halosnap;
  halosnap.Load(world, this_snapnum);
	
  // Update subhalos to the current snapshot
  halosnap.UpdateParticles(world, partsnap);
  subsnap.UpdateParticles(world, partsnap);
  subsnap.AssignHosts(world, halosnap, partsnap);
  subsnap.PrepareCentrals(world, halosnap);
  subsnap.RefineParticles();
  subsnap.MergeSubhalos();
  subsnap.UpdateTracks(world, halosnap);	
  subsnap.Save(world);
}


void hbt_free(void)
{
  // Free HBT state which is stored between outputs
  delete world_ptr;
  if(subsnap_ptr)delete subsnap_ptr;
}
