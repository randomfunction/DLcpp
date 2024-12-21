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

// Gradient Descent class
class GD : public MSE {
private:
    double m; 
    double b;    
    double lr;   
    int epochs;   

public:
    GD(const std::vector<double>& x, const std::vector<double>& y, double m, double b, double lr, int epochs)
        : MSE(x, y), m(m), b(b), lr(lr), epochs(epochs) {}

    void gradientdescent() {
        int n = x.size();
        for (int epoch = 0; epoch < epochs; epoch++) {
            double mGD = 0.0;
            double bGD = 0.0;
            for (int i = 0; i < n; i++) {
                double pred = m * x[i] + b;
                mGD += (pred - y[i]) * x[i];
                bGD += (pred - y[i]);
            }
            mGD /= n;
            bGD /= n;
            m -= lr * mGD;
            b -= lr * bGD;

            if (epoch % 100 == 0) {
                std::cout << "EPOCH " << epoch << " COST " << mse(m, b) << std::endl;
            }
        }
    }

    double getM() const { return m; }
    double getB() const { return b; }
};

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    double m = 0.0;
    double b = 0.0;
    double lr = 0.01;
    int epochs = 1000;

    GD gd1(x, y, m, b, lr, epochs);
    gd1.gradientdescent();
    std::cout << "Final values: m = " << gd1.getM() << ", b = " << gd1.getB() << std::endl;

    return 0;
}
