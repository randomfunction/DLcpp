#include<bits/stdc++.h>
class Adagrad {
private:
    std::vector<double> x; 
    std::vector<double> y;  
    double m;              
    double b;              
    double lr;            
    int epochs;            
    double epsilon;        
    double mAccumulated;    
    double bAccumulated;

public:
    Adagrad(const std::vector<double>& x, const std::vector<double>& y, double m, double b, double lr, int epochs, double epsilon)
        : x(x), y(y), m(m), b(b), lr(lr), epochs(epochs), epsilon(epsilon), mAccumulated(0.0), bAccumulated(0.0) {}

    void gradientDescent() {
        int n = x.size();

        for (int epoch = 0; epoch < epochs; epoch++) {
            double mGradient = 0.0;
            double bGradient = 0.0;
            for (int i = 0; i < n; i++) {
                double pred = m * x[i] + b;
                mGradient += (pred - y[i]) * x[i];
                bGradient += (pred - y[i]);
            }
            mGradient /= n;
            bGradient /= n;
            mAccumulated += mGradient * mGradient;
            bAccumulated += bGradient * bGradient;
            m -= (lr / (std::sqrt(mAccumulated) + epsilon)) * mGradient;
            b -= (lr / (std::sqrt(bAccumulated) + epsilon)) * bGradient;
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
