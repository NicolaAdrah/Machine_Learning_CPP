#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <stdexcept>
#include <stdio.h>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

template <typename T>
class LinearRegression {
  private:
    T m;
    T b;

  public:
  // Define the constructor
    LinearRegression() {
      m = 0.0;
      b = 0.0;
    }

  // Define the fit method
  // this method will take in the x and y values and fit the line to the data based
  // the learning rate and the number of iterations for the gradient descent
  void fit(const std::vector<T>& x, const std::vector<T>& y, T learning_rate, int epochs);
  
  // Define the predict method
  // this method will take in a single x value and return the predicted y value
  T predict(T x){
    return m * x + b;
  }

  // Define the get_slope method
  // this method will return the slope of the line
  T getSlope() const { return m; }

  // Define the get_intercept method
  // this method will return the intercept of the line
  T getIntercept() const { return b; }
};

// Define the fit method
template <typename T>
void LinearRegression<T>::fit(const std::vector<T>& x, const std::vector<T>& y, 
                                                  T learning_rate, int epochs){
  const size_t n = x.size();
  // check if the size of x and y are the same
  if (n == 0 || y.size() != n){
    throw std::invalid_argument("x and y must have the same size");
  }

  T D_m = 0.0; // Derivative of the slope
  T D_b = 0.0; // Derivative of the intercept

  for (int i = 0; i < epochs; i++){
    for (size_t j = 0; j < n; j++) {
      const T prediction = m * x[j] + b;
      const T error = y[j] - prediction;
      D_m += x[j] * error;
      D_b += error;
  }

  // Compute gradients
  D_m *= -2.0 / n;
  D_b *= -2.0 / n;

  // Update weights
  m -= learning_rate * D_m;
  b -= learning_rate * D_b;
  }
}

#endif
