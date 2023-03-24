#pragma once
#include <vector>
#include <functional>

class matrix {
private:
    std::vector<float> data;

public:
    int rows;
    int cols;

    matrix() {};
    matrix(const int matrixRows, const int matrixColumns);
    
    float& operator()(const int row, const int col);

    matrix operator+(matrix& other);

    matrix operator*(const float scalar);
    matrix operator*(matrix& other);

    matrix operator-(matrix& other);
    matrix operator-();

    void applyFunction(std::function<float (float)> f);

    float* getDataPointer();

    void print();
};
