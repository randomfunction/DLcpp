#include<bits/stdc++.h>

class tanH{
    private:
    double x;

    public:
    tanH(double x){
        this->x=x;
    }

    double tanh(){
        return (2/(1+exp(-2*x)))-1;
    }

    double Dtanh(){
        return 1- tanh()*tanh();
    }
};

