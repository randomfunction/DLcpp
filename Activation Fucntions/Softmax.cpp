#include<bits/stdc++.h>

class Softmax{
    private:
    std::vector<double> x;

    public:
    Softmax(std::vector<double> x){
        this->x=x;
    }

    std::vector<double> softmax(){
        std::vector<double> exps(x.size());
        double mx= *std::max_element(x.begin(),x.end());
        double sum=0.0;
        for(size_t i=0;i<x.size();i++){
            exps[i]= std::exp(x[i]- mx);
            sum+=exps[i];
        }
        for(size_t i=0;i<exps.size();i++){
            exps[i]/=sum;
        }
        return exps;
    }

    std::vector<std::vector<double>> Dsoftmax(std::vector<double> &probs){
        size_t n= probs.size();
        std::vector<std::vector<double>> jacobian(n, std::vector<double>(n,0.0));
        for(size_t i=0;i<n;i++){
            for(size_t j=0;j<n;j++){
                if(i==j){
                    jacobian[i][j]= probs[i]*(1-probs[i]);
                }
                else{
                    jacobian[i][j] = -probs[i] * probs[j];
                }
            }
        }
        return jacobian;
    }
};