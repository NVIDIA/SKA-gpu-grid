#ifndef __VIS_H__
#define __VIS_H__ 
#include <iostream>

struct vis
{
  double frequency;
  double time;
  double u;
  double v;
  double w;
  double r;
  double i;
  double weight;
  int a1;
  int a2;
  int gcfinx;
  void print()
  {
     std::cout << "f:" << frequency <<" t:" << time 
               << " (" << u << ", " << v << ", " << w << ") "
               << " => " << r << ", " << i << std::endl;

  }
};
#endif

