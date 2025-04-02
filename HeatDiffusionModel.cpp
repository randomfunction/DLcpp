#include <bits/stdc++.h>
using namespace std;

double energy(double x, double x_opt) {
    return 0.5 * (x - x_opt) * (x - x_opt);
}

double energy_gradient(double x, double x_opt) {
    return (x - x_opt);  
}

// Heat diffusion model simulation
void heat_diffusion_optimization(double x0, double x_opt, double learning_rate, double alpha, double temperature, int iterations, int early_stopping_threshold) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> noise_dist(0.0, 1.0); 

    double x = x0; 
    double best_x = x0; 
    double best_loss = energy(x0, x_opt); 
    int no_improvement_count = 0;

    for (int i = 0; i < iterations; i++) {
        double diffusion = alpha * noise_dist(gen); 
        double grad = energy_gradient(x, x_opt); 
        x = x + diffusion - learning_rate * grad;
        double current_loss = energy(x, x_opt);

        if (current_loss < best_loss) {
            best_loss = current_loss;
            best_x = x;
            no_improvement_count = 0; 
        } else {
            no_improvement_count++;
        }

        cout << "Iteration " << i + 1 << ": x = " << x << ", E(x) = " << energy(x, x_opt) << endl;

        if (no_improvement_count >= early_stopping_threshold) {
            cout << "Early stopping triggered at iteration " << i + 1 << endl;
            break;
        }

        temperature *= 0.99;
    }

    cout << "Final solution: x = " << x << " with energy E(x) = " << energy(x, x_opt) << endl;
}

int main() {
    double x0 = 10.0;          // Initial solution (far from optimum)
    double x_opt = 0.0;        // Optimal solution
    double learning_rate = 0.01;// Step size for gradient descent term
    double alpha = 0.05;       // Diffusion constant (controls how fast heat spreads)
    double temperature = 1.0;  // Initial temperature for cooling
    int iterations = 1000;      // Number of optimization steps
    int early_stopping_threshold = 5;
    
    heat_diffusion_optimization(x0, x_opt, learning_rate, alpha, temperature, iterations,early_stopping_threshold);

    return 0;
}
