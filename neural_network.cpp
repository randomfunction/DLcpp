#include<bits/stdc++.h>
using namespace std;

class ActivationFunction{

    public:

    double binarystep(double x, double binary_threshold){
        return x>=binary_threshold? 1 : 0;
    }

    double sigmoid(double x){
        return 1/(1+ exp(-x));
    }

    double linearActivation(double x){
        return x;
    }

    double tanh(double x){
        return (exp(x)- exp(-x))/(exp(x)+exp(-x));
    }

    double ReLU(double x){
        return max(0.,x);
    }

    double EReLU(double x, double alpha){
        return x>=0?x: alpha*(exp(x)-1);
    }
};