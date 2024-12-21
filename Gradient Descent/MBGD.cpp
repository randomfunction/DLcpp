#include <bits/stdc++.h>
class MSE {
protected:
    std::vector<double> x;
    std::vector<double> y;

public:
    MSE(const std::vector<double>& x, const std::vector<double>& y) : x(x), y(y) {}

    double model(double x, double m, double b) {
        return m * x + b;
    }

    double mse(double m, double b) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            double pred = model(x[i], m, b);
            sum += pow(pred - y[i], 2);
        }
        return sum / x.size();
    }
};

class MiniBatchGD : public MSE {
private:
    double m;   
    double b;      
    double lr;     
    int epochs;    
    int batchSize; 

public:
    MiniBatchGD(const std::vector<double>& x, const std::vector<double>& y, double m, double b, double lr, int epochs, int batchSize)
        : MSE(x, y), m(m), b(b), lr(lr), epochs(epochs), batchSize(batchSize) {}

    void gradientDescent() {
        int n = x.size();
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int batchStart = 0; batchStart < n; batchStart += batchSize) {
                double mGradient = 0.0;
                double bGradient = 0.0;

                // Compute gradients for the current mini-batch
                for (int i = batchStart; i < std::min(batchStart + batchSize, n); i++) {
                    double pred = m * x[i] + b;
                    mGradient += (pred - y[i]) * x[i];
                    bGradient += (pred - y[i]);
                }

                // Average gradients over the mini-batch
                int currentBatchSize = std::min(batchStart + batchSize, n) - batchStart;
                mGradient /= currentBatchSize;
                bGradient /= currentBatchSize;
                m -= lr * mGradient;
                b -= lr * bGradient;
            }
            if (epoch % 100 == 0) {
                std::cout << "EPOCH " << epoch << " COST " << mse(m, b) << std::endl;
            }
        }
    }

    double getM() const { return m; }
    double getB() const { return b; }
};
