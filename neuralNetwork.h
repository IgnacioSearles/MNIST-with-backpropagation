#pragma once
#include "matrix/matrix.h"
#include <vector>
#include <functional>

struct activationFunction {
    std::function<float (const float)> f;
    std::function<float (const float)> fPrime;

    activationFunction() {};
    activationFunction (std::function<float (const float)> activationF, std::function<float (const float)> activationFPrime) {
        f = activationF;
        fPrime = activationFPrime;
    }
};

struct errorFunction {
    std::function<float (const float, const float, const int)> f;
    std::function<float (const float, const float, const int)> fPrime;

    errorFunction() {};
    errorFunction (std::function<float (const float, const float, const int)> errorF, std::function<float (const float, const float, const int)> errorFPrime) {
        f = errorF;
        fPrime = errorFPrime;
    }
};

struct regularizationFunction {
    std::function<float (std::vector<matrix>&, const int, const float)> f;
    std::function<float (const float, const int, const float)> fPrime;

    regularizationFunction() {};
    regularizationFunction (std::function<float (std::vector<matrix>&, const int, const float)> regularizationF, std::function<float (const float, const int, const float)> regularizationFPrime) {
        f = regularizationF;
        fPrime = regularizationFPrime;
    }
};

struct trainingExample {
    matrix input;
    matrix target;

    trainingExample() {};
    trainingExample(matrix trainingInput, matrix trainingTarget) {
        input = trainingInput;
        target = trainingTarget;
    }
};

class neuralNetwork {
    private:
        std::vector<matrix> nodes;
        std::vector<matrix> nodesWithActivation;

        std::vector<matrix> weights;
        std::vector<matrix> biases;

        std::vector<int> shape;

        activationFunction activation;
        errorFunction error;
        regularizationFunction regularization;
        float learningRate;
        float lambda;

        void initParameters();

        static float getRandomNumber(float param);
        void backpropagate(trainingExample& example, std::vector<matrix>& weightsDelta, std::vector<matrix>& biasesDelta);
        void gradientDescent(std::vector<trainingExample>& miniBatch);
        std::vector<matrix> getActivationGradients(matrix& networkOutput, matrix& target);

        static float sigmoidF(const float x);
        static float sigmoidFPrime(const float x);

        static float mseF(const float a, const float t, const int n);
        static float mseFPrime(const float a, const float t, const int n);

        static float L2F(std::vector<matrix>& w, const int n, const float lambda);
        static float L2FPrime(const float w, const int n, const float lambda);
public:
        neuralNetwork(std::vector<int> networkShape, float networkLearningRate = 0.005f, activationFunction networkActivation = sigmoid, errorFunction networkError = mse, regularizationFunction networkRegularization = L2, float regularizationLambda = 0.01f);
        matrix feedfoward(const matrix& input);
        void train(std::vector<trainingExample> examples, int epochs, int miniBatchSize = 5, bool shuffleData = true);

        void save(const char* fileName);
        void load(const char* fileName);

        float getCostOverExamples(std::vector<trainingExample>& examples);
        float getAccuracyOverExamples(std::vector<trainingExample>& examples);
        void print();

        static int oneHotIndex(matrix& out);

        inline static activationFunction sigmoid = activationFunction(sigmoidF, sigmoidFPrime);
        inline static errorFunction mse = errorFunction(mseF, mseFPrime);
        inline static regularizationFunction L2 = regularizationFunction(L2F, L2FPrime);
};
