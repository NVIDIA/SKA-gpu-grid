#include <iostream>
#include <vector>
#include "vis.h"
#include "H5Cpp.h" 

using namespace H5;

struct double2
{
  double x,y;
};

std::vector<struct vis> HDF5_to_struct(H5File* file, int nantenna)
{
   
   std::vector<struct vis> visarray;
   int typeF64 = H5Tcopy(H5T_IEEE_F64LE);
   int typeI64 = H5Tcopy(H5T_STD_I64LE);
   int typeZ = H5Tcreate(H5T_COMPOUND, 16);
   H5Tinsert( (int)typeZ, "r", 0, (int)typeF64);
   H5Tinsert( (int)typeZ, "i", 8, (int)typeF64);
   char str[30];
   char str2[30];
   int gcfinx = 0;
   sprintf(str, "vis/");
   for (int a = 0; a<nantenna;a++)
   for (int b = a+1; b<nantenna;b++,gcfinx++) {
      sprintf(str+4, "%d/", a);
      sprintf(str, "%s%d/", str, b);
      Group groupB = file->openGroup(str);
      sprintf(str2, "%svis/", str);

      //std::cout << str << " has ";
      hsize_t dims_out[3];
      hsize_t offset_out[3] = {0,0,0};
      // Frequency
      DataSet dataset = groupB.openDataSet( H5std_string("frequency"));
      DataSpace space = dataset.getSpace();
      space.getSimpleExtentDims( dims_out, NULL);
      int nfreq = dims_out[0];
      //std::cout << nfreq << " frequencies ";

      double freqs[nfreq];
      
      DataSpace memspace(1, dims_out);
      memspace.selectHyperslab (H5S_SELECT_SET, dims_out, offset_out);
      dataset.read( freqs, typeF64, memspace, space);
      //for (int t=0;t<nfreq;t++) std::cout << freqs[t] << std::endl;

      //Time 
      dataset = groupB.openDataSet( H5std_string("time"));
      space = dataset.getSpace();
      space.getSimpleExtentDims( dims_out, NULL);
      int ntime = dims_out[0];
      //std::cout << "and " << ntime << " timesteps" << std::endl;

      double times[ntime];
      
      memspace.setExtentSimple(1, dims_out);
      memspace.selectHyperslab (H5S_SELECT_SET, dims_out, offset_out);
      dataset.read( times, typeF64, memspace, space);
      //for (int t=0;t<ntime;t++) std::cout << times[t] << std::endl;

      //UVW
      dataset = groupB.openDataSet( H5std_string("uvw"));
      space = dataset.getSpace();
      space.getSimpleExtentDims( dims_out, NULL);

      double uvw[ntime][3];
      
      memspace.setExtentSimple(2, dims_out);
      memspace.selectHyperslab (H5S_SELECT_SET, dims_out, offset_out);
      dataset.read( uvw, typeF64, memspace, space);
      /*for (int t=0;t<ntime;t++) std::cout << uvw[t][0] << ", "
                                          << uvw[t][1] << ", " 
                                          << uvw[t][2] << std::endl;
      */
      
      //weights
      dataset = groupB.openDataSet( H5std_string("weight"));
      space = dataset.getSpace();
      space.getSimpleExtentDims( dims_out, NULL);

      double weights[dims_out[0]][dims_out[1]][1];
      
      memspace.setExtentSimple(3, dims_out);
      memspace.selectHyperslab (H5S_SELECT_SET, dims_out, offset_out);
      dataset.read( weights, typeF64, memspace, space);
      
      //visibilities
      dataset = groupB.openDataSet( H5std_string("vis"));
      space = dataset.getSpace();
      space.getSimpleExtentDims( dims_out, NULL);

      double2 vis[dims_out[0]][dims_out[1]][1];
      
      memspace.setExtentSimple(3, dims_out);
      memspace.selectHyperslab (H5S_SELECT_SET, dims_out, offset_out);
      dataset.read( vis, typeZ, memspace, space);
      for (int t=0;t<dims_out[0];t++)
      { 
         for (int t2=0;t2<dims_out[1];t2++) 
         {
            struct vis vi;
            vi.time = times[t];
            vi.frequency = freqs[t2];
            vi.u = uvw[t][0];
            vi.v = uvw[t][1];
            vi.w = uvw[t][2];
            vi.r = vis[t][t2][0].x;
            vi.i = vis[t][t2][0].y;
            vi.weight = weights[t][t2][0];
            vi.a1 = a;
            vi.a2 = b;
            //TODO Different GCF for each frequency?
            vi.gcfinx = gcfinx;
            //vi.print();
            visarray.push_back(vi);
         }
      }
      
   }
   return visarray;
}

