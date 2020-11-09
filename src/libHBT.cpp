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
static struct libhbt_state_t {

  /* MPI communicator etc */
  MpiWorker_t *world_ptr;

  /* OpenMP threads */
  int num_threads;

  /* Subhalo data to keep between snapshots */
  SubhaloSnapshot_t *subsnap_ptr;

  /* Cosmology */
  double omega_m0, omega_lambda0;

  /* Units of the input data */
  HBTReal MassInMsunh;
  HBTReal LengthInMpch;
  HBTReal VelInKmS;

  /* Group ID which indicates 'not in a group'*/
  HBTInt NullGroupId;
  
} libhbt_state;


extern "C" void hbt_init(char *config_file, int num_threads,
                         double omega_m0, double omega_lambda0,
                         double MassInMsunh, double LengthInMpch,
                         double VelInKmS, long long NullGroupId)
{
  // Store cosmology etc
  libhbt_state.omega_m0 = omega_m0;
  libhbt_state.omega_lambda0 = omega_lambda0;
  libhbt_state.MassInMsunh = MassInMsunh;
  libhbt_state.LengthInMpch = LengthInMpch;
  libhbt_state.VelInKmS = VelInKmS;
  libhbt_state.NullGroupId = NullGroupId;

  // MPI configuration
  libhbt_state.world_ptr = new MpiWorker_t(MPI_COMM_WORLD);
  MpiWorker_t &world = (*libhbt_state.world_ptr);

  // OpenMP configuration
  libhbt_state.num_threads = num_threads;
#ifdef _OPENMP
  omp_set_max_active_levels(1);
  omp_set_num_threads(num_threads);
#endif

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
  libhbt_state.subsnap_ptr = NULL;
}


extern "C" void hbt_invoke(int snapnum, double scalefactor,
                           void *data, size_t np, libhbt_callback_t callback)
{
  MpiWorker_t &world = (*libhbt_state.world_ptr);  
#ifdef _OPENMP
  omp_set_max_active_levels(1);
  omp_set_num_threads(libhbt_state.num_threads);
#endif

  // Read in subhalos from previous snapshot if necessary
  if(!libhbt_state.subsnap_ptr)
    {
      libhbt_state.subsnap_ptr = new SubhaloSnapshot_t;
      (*libhbt_state.subsnap_ptr).Load(world, snapnum-1, SubReaderDepth_t::SrcParticles);
    }
  SubhaloSnapshot_t &subsnap = *libhbt_state.subsnap_ptr;

  // Import particles for this output
  ParticleSnapshot_t partsnap;
  partsnap.Import(world, snapnum, true, scalefactor,
                  libhbt_state.omega_m0, libhbt_state.omega_lambda0,
                  data, np, callback);
  subsnap.SetSnapshotIndex(snapnum);

  // Import halos for this output
  HaloSnapshot_t halosnap;
  halosnap.Import(world, snapnum, scalefactor,
                  libhbt_state.omega_m0, libhbt_state.omega_lambda0,
                  data, np, callback);
	
  // Update subhalos to the current output
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
  delete libhbt_state.world_ptr;
  libhbt_state.world_ptr = NULL;
  if(libhbt_state.subsnap_ptr)delete libhbt_state.subsnap_ptr;
}
