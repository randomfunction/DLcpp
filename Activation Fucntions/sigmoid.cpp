#include<bits/stdc++.h>

class Sigmoid{
    private:
    double x;

    public:
    Sigmoid(double x){
        this->x=x;
    }

    double sigmoid(){
        return 1/(1+exp(-x));
    }

    double Dsigmoid(){
        double num = exp(-x);
        double deno = (1 + exp(-x)) * (1 + exp(-x));
        return num / deno;
    }
};
