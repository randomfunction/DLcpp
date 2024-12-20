#include<iostream>
#include<vector>

class LinearRegression{
    private:
    std::vector<double> y;
    std::vector<double> x;

    public:
    LinearRegression(std::vector<double> xinput, std::vector<double> yinput){
        x= xinput;
        y= yinput;
    }

    void checkSize(){
        int s= std::min(x.size(), y.size());

        x.resize(s);
        y.resize(s);

        std::cout<<x.size()<<" "<<"size of x"<<std::endl;
        std::cout<<y.size()<<" "<<"size of y"<<std::endl;
    }

    double mean(std::vector<double> x){
        double sumx=0;
        for(auto it:x){
            sumx+=it;
        }
        double meanx= sumx/ x.size();
        return meanx;
    }

    double slope(){
        // mean
        double meanx=mean(x);
        double meany=mean(y);

        // sigma(xi-x)(yi-y)
        double num=0;

        for(int i=0;i<x.size();i++){
            num+= (x[i]-meanx)*(y[i]-meany);
        }

        double deno=0;

        for(int i=0;i<x.size();i++){
            deno+= (x[i]-meanx)*(x[i]-meanx);
        }

        double m= num/deno;
        return m;      
    }


    double intercept(){
        return mean(y)- slope()*mean(x);
    }

    std::vector<double> test(std::vector<double> z){
        std::vector<double> zpred(z.size());

        for(int i=0;i<z.size();i++){
            zpred[i]= slope()*z[i]+ intercept();
        }

        return zpred;
    }
};

int main(){
    int n;
    std::cin>>n;
    std::vector<double> x(n);
    std::vector<double> y(n);

    for(int i=0;i<n;i++){
        std::cin>>x[i];
    }

    for(int i=0;i<n;i++){
        std::cin>>y[i];
    }

    LinearRegression LR(x,y);

    LR.checkSize();

    std::vector<double> t(4);
    for(int i=0;i<4;i++){
        std::cin>>t[i];
    }

    std::vector<double> p= LR.test(t);
    for(int i=0;i<4;i++){
        std::cout<<p[i]<<" ";
    }
    std::cout<<std::endl;
}
