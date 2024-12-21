#include <iostream>
#include <vector>
#include <cmath>

class Adam {
private:
    std::vector<double> x;  
    std::vector<double> y;
    double m;             
    double b;             
    double lr;            
    int epochs;          
    double beta1;           // Exponential decay rate for first moment
    double beta2;           // Exponential decay rate for second moment
    double epsilon;         // Small constant for numerical stability

    double m1;              // First moment estimate for slope
    double v1;              // Second moment estimate for slope
    double m2;              // First moment estimate for intercept
    double v2;              // Second moment estimate for intercept
    double t;               // Time step (for bias correction)

public:
    Adam(const std::vector<double>& x, const std::vector<double>& y, double m, double b, double lr, int epochs, double beta1, double beta2, double epsilon)
        : x(x), y(y), m(m), b(b), lr(lr), epochs(epochs), beta1(beta1), beta2(beta2), epsilon(epsilon), m1(0.0), v1(0.0), m2(0.0), v2(0.0), t(0) {}

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
            t++;
            m1 = beta1 * m1 + (1 - beta1) * mGradient;
            v1 = beta2 * v1 + (1 - beta2) * mGradient * mGradient;
            m2 = beta1 * m2 + (1 - beta1) * bGradient;
            v2 = beta2 * v2 + (1 - beta2) * bGradient * bGradient;
            double m1_hat = m1 / (1 - std::pow(beta1, t));  
            double v1_hat = v1 / (1 - std::pow(beta2, t));  
            double m2_hat = m2 / (1 - std::pow(beta1, t)); 
            double v2_hat = v2 / (1 - std::pow(beta2, t));  
            m -= (lr / (std::sqrt(v1_hat) + epsilon)) * m1_hat;
            b -= (lr / (std::sqrt(v2_hat) + epsilon)) * m2_hat;
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

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5,6,7,8};
    std::vector<double> y = {2, 4, 6, 8, 10,12,14,16};
    double m = 0.0;          
    double b = 0.0;       
    double lr = 0.1;       
    int epochs = 1000;       
    double beta1 = 0.9;      
    double beta2 = 0.999;    
    double epsilon = 1e-8;   
    Adam adam(x, y, m, b, lr, epochs, beta1, beta2, epsilon);
    adam.gradientDescent();
    std::cout << "Final values: m = " << adam.getM() << ", b = " << adam.getB() << std::endl;
    return 0;
}
