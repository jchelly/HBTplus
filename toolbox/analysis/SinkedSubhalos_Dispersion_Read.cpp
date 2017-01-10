//CoM,CoV,SigmaR,SigmaV are computed from particle properties while loading subhalos, rather than after.
#include <cmath>
#include <iostream>
#include <string>

#include "../../src/datatypes.h"
#include "../../src/config_parser.h"
#include "../../src/snapshot.h"
#include "../../src/halo.h"
#include "../../src/subhalo.h"
#include "../../src/mymath.h"
#include "../../src/linkedlist_parallel.h"

#define NumPartCore 20
#define DeltaCrit 2. //2 gives high purity, but maybe too agressive? or combine with mass ratio cut?

//TODO: alternatively, use the velocity and position dispersion of the Ncore mostbound particles as scale; or try to rank the binding energy of the sat most-bound in the host, and cut if it's among the N mostbound

struct Satellite_t
{
  HBTInt TrackId;
  HBTInt HostTrackId;
  int TrackDepth;
  float Delta; //Delta to nearest host before sink; Delta to Sink at and after sink
  HBTInt SinkTrackId; //the trackId it sinked to
  int TrackDepthAtSink;//TODO: finish this
  int SinkTrackDepthAtSink;
  float DeltaAtSink;
  float MboundSink;
  float MratSink;
  int SnapshotIndexOfSink;
  HBTxyz CoM, CoV;
  float SigmaR, SigmaV;
  Satellite_t(): 
  TrackId(-1), SinkTrackId(-1), HostTrackId(-1), TrackDepthAtSink(0), SinkTrackDepthAtSink(0), Delta(-1), DeltaAtSink(0), MboundSink(0), MratSink(0), SnapshotIndexOfSink(-1)
  {    }
  bool HasSinked()
  {
    return SnapshotIndexOfSink!=-1;
  }
  hid_t BuildHdfDataType()
  {
    hid_t H5T_dtype=H5Tcreate(H5T_COMPOUND, sizeof (Satellite_t));
    hsize_t dims[2]={3,3};
    hid_t H5T_HBTxyz=H5Tarray_create2(H5T_HBTReal, 1, dims);
    
    #define InsertMember(x,t) H5Tinsert(H5T_dtype, #x, HOFFSET(Satellite_t, x), t)//;cout<<#x<<": "<<HOFFSET(HaloSize_t, x)<<endl
    InsertMember(TrackId, H5T_HBTInt);
    InsertMember(HostTrackId, H5T_HBTInt);
    InsertMember(TrackDepth, H5T_NATIVE_INT);
    InsertMember(Delta, H5T_NATIVE_FLOAT);
    InsertMember(SinkTrackId, H5T_HBTInt);
    InsertMember(TrackDepthAtSink, H5T_NATIVE_INT);
    InsertMember(SinkTrackDepthAtSink, H5T_NATIVE_INT);
    InsertMember(DeltaAtSink, H5T_NATIVE_FLOAT);
    InsertMember(MboundSink, H5T_NATIVE_FLOAT);
    InsertMember(MratSink, H5T_NATIVE_FLOAT);
    InsertMember(SnapshotIndexOfSink, H5T_NATIVE_INT);
    InsertMember(CoM, H5T_HBTxyz);
    InsertMember(CoV, H5T_HBTxyz);
    InsertMember(SigmaR, H5T_NATIVE_FLOAT);
    InsertMember(SigmaV, H5T_NATIVE_FLOAT);
    #undef InsertMember
    
    H5Tclose(H5T_HBTxyz);
    return H5T_dtype;
  }
};
void FillDepth(HBTInt subid, int depth, vector <Subhalo_t> &Subhalos, vector <Satellite_t> &Satellites)
{
  Satellites[subid].TrackDepth=depth;
  depth++;
  for(auto &&nestid: Subhalos[subid].NestedSubhalos)
  {
    FillDepth(nestid, depth, Subhalos, Satellites);
  }
}

void save(vector <Satellite_t> &Satellites, int isnap);
double PositionStat(HBTxyz& CoM, const vector <HBTInt> & PartIndex, const ParticleSnapshot_t &snap);
double VelocityStat(HBTxyz& CoV, const vector <HBTInt> & PartIndex, const ParticleSnapshot_t &snap);
float SinkDistance(Subhalo_t &sat, Satellite_t &cen)
{
  float d=PeriodicDistance(cen.CoM, sat.ComovingMostBoundPosition);
  float v=Distance(cen.CoV, sat.PhysicalMostBoundVelocity);
  return d/cen.SigmaR+v/cen.SigmaV;
}
int main(int argc,char **argv)
{
  int snapshot_start, snapshot_end;
  ParseHBTParams(argc, argv, HBTConfig, snapshot_start, snapshot_end);
  
  vector <Satellite_t> Satellites;
  for(int isnap=snapshot_start;isnap<=snapshot_end;isnap++)
  {
    SubhaloSnapshot_t subsnap(isnap);
    auto &SubGroups=subsnap.MemberTable.SubGroups;
    HBTInt nold=Satellites.size();
    Satellites.resize(subsnap.Subhalos.size());
    
    #pragma omp parallel
    {
      #pragma omp for
      for(HBTInt i=nold;i<Satellites.size();i++)
	Satellites[i].TrackId=i;
      //reinit host
      #pragma omp for
      for(HBTInt i=0;i<Satellites.size();i++) 
      {
	Satellites[i].HostTrackId=-1;
	Satellites[i].Delta=-1.;//reinit
	Satellites[i].SigmaR=subsnap.Subhalos[i].ComovingCoreSigmaR;
	Satellites[i].SigmaV=subsnap.Subhalos[i].PhysicalCoreSigmaV;
	copyHBTxyz(Satellites[i].CoM, subsnap.Subhalos[i].ComovingCorePosition);
	copyHBTxyz(Satellites[i].CoV, subsnap.Subhalos[i].PhysicalCoreVelocity);
      }
      #pragma omp for
      for(HBTInt i=0;i<Satellites.size();i++) 
      {
	auto &nest=subsnap.Subhalos[i].NestedSubhalos;
	for(auto &&subid: nest)
	  Satellites[subid].HostTrackId=i;
      }
      
      #pragma omp for
      for(HBTInt grpid=0;grpid<SubGroups.size();grpid++)
	if(SubGroups[grpid].size()) FillDepth(SubGroups[grpid][0], 0, subsnap.Subhalos, Satellites);
	
	#pragma omp  for
	for(HBTInt i=0;i<Satellites.size();i++)
	{
	  if(Satellites[i].HasSinked())
	  {
	    Satellites[i].Delta=SinkDistance(subsnap.Subhalos[i], Satellites[Satellites[i].SinkTrackId]);
	    continue;
	  }
	  HBTInt HostId=Satellites[i].HostTrackId;
	  float deltaMin=-1.;
	  HBTInt SinkId=HostId;
	  while(HostId>=0)
	  {
	    float delta=SinkDistance(subsnap.Subhalos[i], Satellites[HostId]);
	    if(delta<deltaMin||deltaMin<0) 
	    {
	      deltaMin=delta;
	      SinkId=HostId;
	    } 
// 	    if(Satellites[i].Delta<0) Satellites[i].Delta=delta;
	    if(delta<DeltaCrit)
	    {
	      Satellites[i].Delta=delta;
	      Satellites[i].SinkTrackId=HostId;
	      Satellites[i].TrackDepthAtSink=Satellites[i].TrackDepth;
	      Satellites[i].SinkTrackDepthAtSink=Satellites[HostId].TrackDepth;
	      Satellites[i].DeltaAtSink=delta;
	      Satellites[i].MboundSink=subsnap.Subhalos[i].Mbound;
	      Satellites[i].MratSink=subsnap.Subhalos[i].Mbound/subsnap.Subhalos[HostId].Mbound;
	      Satellites[i].SnapshotIndexOfSink=isnap;
	      break;
	    }
	    HostId=Satellites[HostId].HostTrackId;
	  }
	  if(!Satellites[i].HasSinked()&&SinkId>=0) //bind to the minimum delta before sink
	  {
	    Satellites[i].Delta=deltaMin;
	    Satellites[i].SinkTrackId=SinkId;
	  }
	}
    }
    
    save(Satellites, isnap);
  }
  
  return 0;
}

void save(vector <Satellite_t> &Satellites, int isnap)
{
  string outdir=HBTConfig.SubhaloPath+"/Sink_Dispersion"+to_string(int(DeltaCrit))+".test/";
  mkdir(outdir.c_str(),0755);
  stringstream filename;
  filename<<outdir<<"SinkSnap_"<<setw(3)<<setfill('0')<<isnap<<".hdf5";
  hid_t file=H5Fcreate(filename.str().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dtype=Satellite_t().BuildHdfDataType();
  hsize_t dims[]= {Satellites.size()};
  writeHDFmatrix(file, Satellites.data(), "Subhalos", 1, dims, dtype);
  H5Tclose(dtype);
  H5Fclose(file);
}

double PositionStat(HBTxyz& CoM, const vector<HBTInt>& PartIndex, const ParticleSnapshot_t &snap)
/*return radial dispersion*/
{
  HBTInt NumPart=PartIndex.size();
  if(NumPart>NumPartCore) NumPart=NumPartCore;
  
  HBTInt i,j;
  double sx[3],sx2[3], origin[3],msum;
  
  if(0==NumPart) return 0.;
  if(1==NumPart) 
  {
    copyHBTxyz(CoM, snap.GetComovingPosition(PartIndex[0]));
    return 0.;
  }
  
  sx[0]=sx[1]=sx[2]=0.;
  sx2[0]=sx2[1]=sx2[2]=0.;
  msum=0.;
  if(HBTConfig.PeriodicBoundaryOn)
    for(j=0;j<3;j++)
      origin[j]=snap.GetComovingPosition(PartIndex[0])[j];
    
    for(i=0;i<NumPart;i++)
    {
      HBTReal m=snap.GetParticleMass(PartIndex[i]);
      msum+=m;
      for(j=0;j<3;j++)
      {
	double dx;
	if(HBTConfig.PeriodicBoundaryOn)
	  dx=NEAREST(snap.GetComovingPosition(PartIndex[i])[j]-origin[j]);
	else
	  dx=snap.GetComovingPosition(PartIndex[i])[j];
	sx[j]+=dx*m;
	sx2[j]+=dx*dx*m;
      }
    }
    
    for(j=0;j<3;j++)
    {
      sx[j]/=msum;
      sx2[j]/=msum;
      CoM[j]=sx[j];
      if(HBTConfig.PeriodicBoundaryOn) CoM[j]+=origin[j];
      sx2[j]-=sx[j]*sx[j];
    }
    return sqrt(sx2[0]+sx2[1]+sx2[2]);
}
double VelocityStat(HBTxyz& CoV, const vector<HBTInt> &PartIndex,  const ParticleSnapshot_t &snap)
/*return velocity dispersion*/
{
  HBTInt NumPart=PartIndex.size();
  if(NumPart>NumPartCore) NumPart=NumPartCore;
  
  HBTInt i,j;
  double sx[3],sx2[3],msum;
  
  if(0==NumPart) return 0.;
  if(1==NumPart) 
  {
    copyHBTxyz(CoV, snap.GetPhysicalVelocity(PartIndex[0]));
    return 0.;
  }
  
  sx[0]=sx[1]=sx[2]=0.;
  sx2[0]=sx2[1]=sx2[2]=0.;
  msum=0.;
  
  for(i=0;i<NumPart;i++)
  {
    HBTReal m=snap.GetParticleMass(PartIndex[i]);
    msum+=m;
    for(j=0;j<3;j++)
    {
      double dx;
      dx=snap.GetPhysicalVelocity(PartIndex[i])[j];
      sx[j]+=dx*m;
      sx2[j]+=dx*dx*m;
    }
  }
  
  for(j=0;j<3;j++)
  {
    sx[j]/=msum;
    sx2[j]/=msum;
    CoV[j]=sx[j];
    sx2[j]-=sx[j]*sx[j];
  }
  return sqrt(sx2[0]+sx2[1]+sx2[2]);
}
