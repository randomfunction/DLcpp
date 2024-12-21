#include<bits/stdc++.h>

class AcceleratedGD {
private:
    std::vector<double> x;  
    std::vector<double> y; 
    double m;            
    double b;           
    double lr;            
    int epochs;           
    double gamma;         

public:
    AcceleratedGD(const std::vector<double>& x, const std::vector<double>& y, double m, double b, double lr, int epochs, double gamma)
        : x(x), y(y), m(m), b(b), lr(lr), epochs(epochs), gamma(gamma) {}

    void gradientDescent() {
        double mVelocity = 0.0;
        double bVelocity = 0.0; 

        for (int epoch = 0; epoch < epochs; epoch++) {
            double mGradient = 0.0;
            double bGradient = 0.0;
            int n = x.size();
            for (int i = 0; i < n; i++) {
                double pred = m * x[i] + b;
                mGradient += (pred - y[i]) * x[i];
                bGradient += (pred - y[i]);
            }
            mGradient /= n;
            bGradient /= n;
            mVelocity = gamma * mVelocity + lr * mGradient;
            bVelocity = gamma * bVelocity + lr * bGradient;
            m -= mVelocity;
            b -= bVelocity;
            if (epoch % 100 == 0) {
                double cost = 0.0;
                for (int i = 0; i < n; i++) {
                    double pred = m * x[i] + b;
                    cost += pow(pred - y[i], 2);
                }
                cost /= n;
                std::cout << "EPOCH " << epoch << " COST " << cost << std::endl;
            }
        }
    }

    double getM() const { return m; }
    double getB() const { return b; }
};