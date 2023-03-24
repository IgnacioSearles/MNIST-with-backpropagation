#include "matrix.h"
#include <iostream>
#include <stdexcept>

matrix::matrix(const int matrixRows, const int matrixColumns) {
    rows = matrixRows;
    cols = matrixColumns;

    for (int i = 0; i < rows * cols; i++) data.push_back(0);
}

float& matrix::operator()(const int row, const int col) {
    if (row * cols + col >= data.size()) throw std::logic_error("Value out of range");
    return data[row * cols + col];
}

matrix matrix::operator+(matrix& other) {
    if (rows != other.rows || cols != other.cols) throw std::logic_error("Can't add matrices of different dimensions");

    matrix out(rows, cols);

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            out(row, col) = (*this)(row, col) + other(row, col);
        }
    }

    return out;
}

matrix matrix::operator*(const float scalar) {
     matrix out(rows, cols);

    for (int row = 0; row < rows; row++)
        for (int col = 0; col < cols; col++)
            out(row, col) = (*this)(row, col) * scalar;

    return out;
}


matrix matrix::operator*(matrix& other) {
    if (cols != other.rows) throw std::logic_error("Can't multiply matrices of m1.cols != m2.rows");

    matrix out(rows, other.cols);

    for (int row = 0; row < out.rows; row++)
        for (int col = 0; col < out.cols; col++)
            for (int k = 0; k < cols; k++) 
                out(row, col) += (*this)(row, k) * other(k, col);

    return out;
}

matrix matrix::operator-(matrix& other) {
    return (other * -1.0f) + (*this);
}

matrix matrix::operator-() {
    return (*this) * -1.0f;
}

void matrix::applyFunction(std::function<float (float)> f) {
    for (int i = 0; i < rows * cols; i++) data[i] = f(data[i]);
}

float* matrix::getDataPointer() {
    return data.data();
}

void matrix::print() {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++)
            std::cout << data[row * cols + col] << " ";
        std::cout << std::endl;
    }
}
