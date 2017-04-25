#include "mymath.h"
#include "linkedlist.h"

HBTReal Linkedlist_t::Distance(const HBTxyz &x, const HBTxyz &y)
{
  HBTxyz dx;
  dx[0]=x[0]-y[0];
  dx[1]=x[1]-y[1];
  dx[2]=x[2]-y[2];
  if(PeriodicBoundary)
  {
    #define _NEAREST(x) (((x)>BoxHalf)?((x)-BoxSize):(((x)<-BoxHalf)?((x)+BoxSize):(x)))
    dx[0]=_NEAREST(dx[0]);
    dx[1]=_NEAREST(dx[1]);
    dx[2]=_NEAREST(dx[2]);
    #undef _NEAREST
  }
  return sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
}
  
void Linkedlist_t::build(int ndiv, PositionData_t *data, HBTReal boxsize, bool periodic) 
{
  NDiv=ndiv;
  NDiv2=NDiv*NDiv;
  Particles=data;
  PositionData_t &particles=*Particles;
  HBTInt np=particles.size();
  HOC.assign(NDiv2*NDiv, -1);
  List.resize(np);
  BoxSize=boxsize;
  PeriodicBoundary=periodic;
  if(BoxSize==0.) PeriodicBoundary=false; //only effective when boxsize is specified
  BoxHalf=BoxSize/2.;
  
  
  HBTInt i,j,grid[3];
  HBTInt ind;
  //~ float range[3][2],step[3];
  cout<<"creating linked list for "<<particles.size()<<" particles..."<<endl;
  /*determining enclosing cube*/
  if(BoxSize)
  {
    for(i=0;i<3;i++)
    {
      Range[i][0]=0.;
      Range[i][1]=BoxSize;
    }
    for(j=0;j<3;j++)
      Step[j]=BoxSize/NDiv;	
  }
  else
  {
    for(i=0;i<3;i++)
      for(j=0;j<2;j++)
	Range[i][j]=particles[0][j];
    for(i=1;i<np;i++)
      for(j=0;j<3;j++)
      {
	auto x=particles[i][j];
	if(x<Range[j][0])
	  Range[j][0]=x;
	else if(x>Range[j][1])
	  Range[j][1]=x;
      }
    for(j=0;j<3;j++)
      Step[j]=(Range[j][1]-Range[j][0])/NDiv;
  }
	  
  for(i=0;i<np;i++)
  {
	  for(j=0;j<3;j++)
	  {
		  grid[j]=floor((particles[i][j]-Range[j][0])/Step[j]);
		  grid[j]=FixGridId(grid[j]);
	  }
	  ind=Sub2Ind(grid[0],grid[1],grid[2]);
	  List[i]=HOC[ind];
	  HOC[ind]=i; /*use hoc[ind] as swap varible to temporarily 
								  store last ll index, and finally the head*/
  }
}
void Linkedlist_t::SearchSphere(HBTReal radius, const HBTxyz &searchcenter, vector <LocatedParticle_t> &founds, int nmax_guess, HBTReal rmin)
{//nmax_guess: initial guess for the max number of particles to be found. for memory allocation optimization purpose.
  PositionData_t &particles=*Particles;
  HBTReal dr;
  int i,j,k,subbox_grid[3][2];
  
  founds.clear();
  founds.reserve(nmax_guess);
	  
  for(i=0;i<3;i++)
  {
    subbox_grid[i][0]=floor((searchcenter[i]-radius-Range[i][0])/Step[i]);
    subbox_grid[i][1]=floor((searchcenter[i]+radius-Range[i][0])/Step[i]);
    if(!PeriodicBoundary)
    {//do not fix if periodic, since the search sphere is allowed to overflow the box in periodic case.
      subbox_grid[i][0]=FixGridId(subbox_grid[i][0]);
      subbox_grid[i][1]=FixGridId(subbox_grid[i][1]);
    }	
  }
  for(i=subbox_grid[0][0];i<subbox_grid[0][1]+1;i++)
    for(j=subbox_grid[1][0];j<subbox_grid[1][1]+1;j++)
      for(k=subbox_grid[2][0];k<subbox_grid[2][1]+1;k++)
      {
	HBTInt pid=GetHOCSafe(i,j,k); //in case the grid-id is out of box, in the periodic case
	while(pid>=0)
	{
	  dr=Distance(particles[pid],searchcenter);
	  if(dr<radius&&dr>rmin)  founds.push_back(LocatedParticle_t(pid,dr));
	  pid=List[pid];
	}
      }
}
