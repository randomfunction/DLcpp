#include <iostream>
#include <vector>
#include <stdexcept>

class LinearRegression {
private:
    std::vector<double> y;
    std::vector<std::vector<double>> x;

public:
    LinearRegression(const std::vector<std::vector<double>> &xinput, const std::vector<double> &yinput) {
        x = xinput; // size = n*m
        y = yinput; // size= n*1
    }

    void checkSize() {
        std::cout << "x size: " << x.size() << " x[0] size: " << x[0].size() << std::endl;
        std::cout << "y size: " << y.size() << std::endl;
    }

    void prepareX() {
        int n = x.size();
        for (int i = 0; i < n; i++) {
            x[i].insert(x[i].begin(), 1); // Add the bias term
        }
    }

    // Transpose of a matrix
    std::vector<std::vector<double>> Transpose(const std::vector<std::vector<double>> &mat) {
        std::vector<std::vector<double>> transposed(mat[0].size(), std::vector<double>(mat.size()));
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < mat[0].size(); ++j) {
                transposed[j][i] = mat[i][j];
            }
        }
        return transposed;
    }

    // Matrix multiplication
    std::vector<std::vector<double>> matMultiply(const std::vector<std::vector<double>> &A, const std::vector<std::vector<double>> &B) {
        size_t n = A.size(), m = B[0].size(), k = A[0].size();
        std::vector<std::vector<double>> result(n, std::vector<double>(m, 0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t l = 0; l < k; ++l) {
                    result[i][j] += A[i][l] * B[l][j];
                }
            }
        }
        return result;
    }

    // Matrix inversion (Gaussian elimination)
    std::vector<std::vector<double>> inverse(const std::vector<std::vector<double>> &mat) {
        int n = mat.size();
        std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i][j] = mat[i][j];
            }
            augmented[i][n + i] = 1;
        }

        for (int i = 0; i < n; ++i) {
            if (augmented[i][i] == 0) {
                int swapRow = -1;
                for (int j = i + 1; j < n; ++j) {
                    if (augmented[j][i] != 0) {
                        swapRow = j;
                        break;
                    }
                }
                if (swapRow == -1) {
                    throw std::runtime_error("Matrix is singular and cannot be inverted.");
                }
                std::swap(augmented[i], augmented[swapRow]);
            }

            double diagElement = augmented[i][i];
            for (int j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= diagElement;
            }

            for (int k = 0; k < n; ++k) {
                if (k == i) continue;
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }

        std::vector<std::vector<double>> inverse(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inverse[i][j] = augmented[i][n + j];
            }
        }
        return inverse;
    }

    // Compute theta = (X^T * X)^(-1) * X^T * y
    std::vector<double> computeTheta() {
        prepareX();
        auto xT = Transpose(x);
        auto xTx = matMultiply(xT, x);
        auto xTx_inv = inverse(xTx);
        auto xTy = matMultiply(xT, {y});
        auto thetaMat = matMultiply(xTx_inv, xTy);

        std::vector<double> theta(thetaMat.size());
        for (size_t i = 0; i < theta.size(); ++i) {
            theta[i] = thetaMat[i][0];
        }
        return theta;
    }

    std::vector<double> test(const std::vector<double> &features) {
        std::vector<double> testFeatures = features;
        testFeatures.insert(testFeatures.begin(), 1); 
        auto theta = computeTheta();
        double prediction = 0.0;

        for (size_t i = 0; i < testFeatures.size(); ++i) {
            prediction += theta[i] * testFeatures[i];
        }
        std::cout<<"isitworking"<<std::endl;
        return {prediction};
    }
};

int main() {
    int n, m;
    std::cin >> n >> m;
    std::vector<std::vector<double>> x(n, std::vector<double>(m));
    std::vector<double> y(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cin >> x[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        std::cin >> y[i];
    }

    LinearRegression LR(x, y);

    LR.checkSize();

    std::vector<double> t(m);
    for (int i = 0; i < m; ++i) {
        std::cin >> t[i];
    }
    std::vector<double> p = LR.test(t);
    std::cout << "Prediction: " << p[0] << std::endl;

    return 0;
}
