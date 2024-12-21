#include<bits/stdc++.h>

class ReLU{
    private:
    double x;
    double slope;

    public:
    ReLU(double x, double slope){
        this->x=x;
    }

    double relu(){
        return std::max(slope*x,x);
    }

    double DReLU(){
        return x<0? slope:1;
    }
};

