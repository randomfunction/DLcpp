#include<bits/stdc++.h>

class MSE{
    private:
    std::vector<double> test;
    std::vector<double> x;

    public:
    MSE(std::vector<double> test, std::vector<double> x){
        this->test=test;
        this->x=x;
    } 

    double model(double x){};

    double mse(){
        double sum=0.0;
        for(size_t i=0;i<test.size();i++){
            double pred= model(x[i]);
            sum+=pow(pred- test[i], 2);
        }
        return sum/test.size();
    }
};


// simple linear regression example 
class GD{
    private:
    std::vector<double> x;
    std::vector<double> y;
    double m;
    double b;
    double lr;
    int epochs;

    public:
    GD(std::vector<double> x, std::vector<double> y, double m, double b, double lr, int epochs){
        this->x=x;
        this->y=y;
        this->m=m;
        this->b=b;
        this->lr=lr;
        this->epochs= epochs;
    }

    


};