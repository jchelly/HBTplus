using namespace std;
#include <iostream>
// #include <iomanip>
#include <sstream>
#include <string>
#include <typeinfo>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <mpi.h>

#include "snapshot.h"
#include "mymath.h"


#ifdef HBT_LIBRARY
void ParticleSnapshot_t::Import(MpiWorker_t &world, int snapshot_index, bool fill_particle_hash,
    double scalefactor, double omega_m0, double omega_lambda0,
    void *data, size_t np, libhbt_callback_t callback)
{
  Clear();
  SetSnapshotIndex(snapshot_index);
  Cosmology.Set(scalefactor, omega_m0, omega_lambda0);
  Particles.resize(np);

  // Loop over particles to import
  for(size_t i=0; i<np; i++) {

    // Fetch information about this particle
    HBTInt  fofid;
    HBTInt  type;
    HBTReal pos[3];
    HBTReal vel[3];
    HBTInt  id;
    HBTReal mass;
    HBTReal u;
    (*callback)(data, i, &fofid, &type, pos, vel, &id, &mass, &u);
    
    // Coordinates: box wrap if necessary
    if(HBTConfig.PeriodicBoundaryOn)
      for(int j=0;j<3;j++)
        pos[j]=position_modulus(pos[j], HBTConfig.BoxSize);
    for(int j=0;j<3;j++)
      Particles[i].ComovingPosition[j] = pos[j];

    // Velocity
    for(int j=0;j<3;j++)
      Particles[i].PhysicalVelocity[j]=vel[j];

    // ID
    Particles[i].Id=id;

    // Mass
    Particles[i].Mass=mass;

#ifndef DM_ONLY
#ifdef HAS_THERMAL_ENERGY
    // Internal energy
    Particles[i].InternalEnergy=u;
#endif
    // Type
    ParticleType_t t=static_cast<ParticleType_t>(type);
    Particles[i].Type=t;
#else
    if(type!=1) {
      cout << "HBT compiled with DM_ONLY but have particle with type!=1" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
  }

  ExchangeParticles(world);
  
  if(fill_particle_hash)
    FillParticleHash();
  
  if(world.rank()==0) cout<<NumberOfParticlesOnAllNodes<<" particles imported at Snapshot "<<snapshot_index<<"("<<SnapshotId<<")"<<endl;
}
#endif
