#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <typeinfo>
#include <assert.h>
#include <glob.h>
#include <algorithm>
#include <chrono>

#include "mpi_wrapper.h"
#include "mymath.h"
#include "halo.h"
#include "particle_exchanger.h"

struct ImportParticleHost_t: public Particle_t
{
  HBTInt HostId;
};

#ifdef HBT_LIBRARY
void HaloSnapshot_t::Import(MpiWorker_t &world, int snapshot_index,
                            bool fill_particle_hash, double scalefactor,
                            double omega_m0, double omega_lambda0,
                            void *data, size_t np, libhbt_callback_t callback)
{
  SetSnapshotIndex(snapshot_index);
  


  /*
  string GroupFileFormat=HBTConfig.GroupFileFormat;
  if(GadgetGroup::IsGadgetGroup(GroupFileFormat))
	GadgetGroup::Load(world, SnapshotId, Halos);
  else if(IsApostleGroup(GroupFileFormat))
	ApostleReader_t().LoadGroups(world, SnapshotId, Halos);
  else if(IsSwiftSimGroup(GroupFileFormat))
	SwiftSimReader_t().LoadGroups(world, SnapshotId, Halos);
  else if(GroupFileFormat=="my_group_format")
  {

  }
  else
	throw(runtime_error("unknown GroupFileFormat "+GroupFileFormat));
  */

  NumPartOfLargestHalo=0;
  TotNumberOfParticles=0;
  for(auto && h: Halos)
  {
    auto np=h.Particles.size();
    TotNumberOfParticles+=np;
    if(np>NumPartOfLargestHalo) NumPartOfLargestHalo=np;
  }
  
  HBTInt NumHalos=Halos.size(), NumHalosAll=0;
  MPI_Reduce(&NumHalos, &NumHalosAll, 1, MPI_HBT_INT, MPI_SUM, 0, world.Communicator);
  if(world.rank()==0)
    cout<<NumHalosAll<<" groups loaded at snapshot "<<snapshot_index<<"("<<SnapshotId<<")"<<endl;
}
#endif
