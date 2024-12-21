#include<bits/stdc++.h>

class ReLU{
    private:
    double x;

    public:
    ReLU(double x){
        this->x=x;
    }

    double relu(){
        return std::max(0.0,x);
    }

    double DReLU(){
        return x<0? 0:1;
    }
};

