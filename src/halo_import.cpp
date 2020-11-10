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
#include "io/exchange_and_merge.h"

#ifdef HBT_LIBRARY
struct ImportParticleHost_t: public Particle_t
{
  HBTInt HostId;
};

inline bool CompImportParticleHost(const ImportParticleHost_t &a, const ImportParticleHost_t &b)
{
  return a.HostId<b.HostId;
}

void HaloSnapshot_t::Import(MpiWorker_t &world, int snapshot_index,
                            double scalefactor, double omega_m0, double omega_lambda0,
                            HBTInt NullGroupId, void *data, size_t np, libhbt_callback_t callback)
{
  SetSnapshotIndex(snapshot_index);

  // Set up particle vector
  vector<ImportParticleHost_t> ParticleHosts;
  ParticleHosts.resize(np);

  // Loop over particles to import
#pragma omp parallel for
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
      ParticleHosts[i].ComovingPosition[j] = pos[j];

    // Velocity
    for(int j=0;j<3;j++)
      ParticleHosts[i].PhysicalVelocity[j]=vel[j];

    // ID
    ParticleHosts[i].Id=id;

    // Mass
    ParticleHosts[i].Mass=mass;

#ifndef DM_ONLY
#ifdef HAS_THERMAL_ENERGY
    // Internal energy
    ParticleHosts[i].InternalEnergy=u;
#endif
    // Type
    ParticleType_t t=static_cast<ParticleType_t>(type);
    ParticleHosts[i].Type=t;
#else
    if(type!=1) {
      cout << "HBT compiled with DM_ONLY but have particle with type!=1" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif

    // FoF group index
    ParticleHosts[i].HostId=fofid;
  }

  // Construct halos
  sort(ParticleHosts.begin(), ParticleHosts.end(), CompImportParticleHost);
  if(!ParticleHosts.empty())
  {
    assert(ParticleHosts.back().HostId<=NullGroupId);//max haloid==NullGroupId
    assert(ParticleHosts.front().HostId>=0);//min haloid>=0
  }
  
  struct HaloLen_t
  {
    HBTInt haloid;
    HBTInt np;
    HaloLen_t(){};
    HaloLen_t(HBTInt haloid, HBTInt np): haloid(haloid), np(np)
    {
    }
  };
  vector <HaloLen_t> HaloLen;
  
  HBTInt curr_host_id=NullGroupId;
  for(auto &&p: ParticleHosts)
  {
    // TODO: remove this assumption in case NullGroupId is not a large value?
    if(p.HostId==NullGroupId) break;//NullGroupId comes last
    if(p.HostId!=curr_host_id)
    {
      curr_host_id=p.HostId;
      HaloLen.emplace_back(curr_host_id, 1);
    }
    else
      HaloLen.back().np++;
  }
  Halos.resize(HaloLen.size());
  for(HBTInt i=0;i<Halos.size();i++)
  {
    Halos[i].HaloId=HaloLen[i].haloid;
    Halos[i].Particles.resize(HaloLen[i].np);
  }
  auto p_in=ParticleHosts.begin();
  for(auto &&h: Halos)
  {
    for(auto &&p: h.Particles)
    {
      p=*p_in;
      ++p_in;
    }
  }
  
  VectorFree(ParticleHosts);
  
  ExchangeAndMerge(world, Halos);

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
    cout<<NumHalosAll<<" groups imported at snapshot "<<snapshot_index<<"("<<SnapshotId<<")"<<endl;
}
#endif
