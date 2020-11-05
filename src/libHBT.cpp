using namespace std;
#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>

#include "mpi_wrapper.h"
#include "datatypes.h"
#include "config_parser.h"
#include "snapshot.h"
#include "halo.h"
#include "subhalo.h"
#include "mymath.h"
#include "particle_exchanger.h"

// Include header to check consistency with code
#include "libHBT.h"

// Data which must persist between HBT invocations
static MpiWorker_t *world_ptr;
static int num_hbt_threads;
static SubhaloSnapshot_t *subsnap_ptr;


extern "C" void hbt_init(char *config_file, int num_threads)
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


extern "C" void hbt_invoke(int snapnum, double scalefactor,
                           double omega_m0, double omega_lambda0,
                           void *data, size_t np, libhbt_callback_t callback)
{
  MpiWorker_t &world = (*world_ptr);  
#ifdef _OPENMP
  omp_set_max_active_levels(1);
  omp_set_num_threads(num_hbt_threads);
#endif

  // Read in subhalos from previous snapshot if necessary
  if(!subsnap_ptr)
    {
      subsnap_ptr = new SubhaloSnapshot_t;
      (*subsnap_ptr).Load(world, snapnum-1, SubReaderDepth_t::SrcParticles);
    }
  SubhaloSnapshot_t &subsnap = (*subsnap_ptr);

  // Import particles and halos for this output
  ParticleSnapshot_t partsnap;
  partsnap.Import(world, snapnum, true,
                  scalefactor, omega_m0, omega_lambda0,
                  data, np, callback);
  subsnap.SetSnapshotIndex(snapnum);
  HaloSnapshot_t halosnap;
  halosnap.Load(world, snapnum);
	
  // Update subhalos to the current snapshot
  halosnap.UpdateParticles(world, partsnap);
  subsnap.UpdateParticles(world, partsnap);
  subsnap.AssignHosts(world, halosnap, partsnap);
  subsnap.PrepareCentrals(world, halosnap);
  if(world.rank()==0) cout<<"unbinding...\n";
  subsnap.RefineParticles();
  subsnap.MergeSubhalos();
  subsnap.UpdateTracks(world, halosnap);	
  subsnap.Save(world);
}


extern "C" void hbt_free(void)
{
  // Free HBT state which is stored between outputs
  delete world_ptr;
  if(subsnap_ptr)delete subsnap_ptr;
}
