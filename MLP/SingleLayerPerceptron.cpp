#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib> 
#include <ctime>  

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double Dsigmoid(double x) {
    double num = exp(-x);
    double deno = (1 + exp(-x)) * (1 + exp(-x));
    return num / deno;
}

class Perceptron {
private:
    std::vector<double> w; 
    double b;              
    double lr;            

public:
    Perceptron(int inputsize, double lr) {
        this->lr = lr; 
        srand(time(0)); 

        for (int i = 0; i < inputsize; i++) {
            w.push_back(((double)rand() / RAND_MAX) * 2 - 1); 
        }
        b = ((double)rand() / RAND_MAX) * 2 - 1; 
    }

    double forward(std::vector<double> inputs) {
        double Wsum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            Wsum += inputs[i] * w[i];
        }
        Wsum += b;
        return sigmoid(Wsum);
    }

    void train(std::vector<std::vector<double>> x, std::vector<double> y, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double Terror = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double prediction = forward(x[i]);
                double error = y[i] - prediction;
                Terror += error * error;

                double gradient = error * Dsigmoid(prediction);
                for (int j = 0; j < w.size(); j++) {
                    w[j] += lr * gradient * x[i][j];
                }
                b += lr * gradient;
            }

            std::cout << "Epoch " << epoch + 1 << ", Error: " << Terror / x.size() << std::endl;
        }
    }
};

int main() {
    std::vector<std::vector<double>> x = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y = {0, 1, 1, 0};
    Perceptron model(2, 0.1);
    model.train(x, y, 100);
    std::cout << "Testing:" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        double output = model.forward(x[i]);
        std::cout << "Inputs: ";
        for (double val : x[i]) std::cout << val << " ";
        std::cout << "-> Output: " << output << std::endl;
    }

    return 0;
}
