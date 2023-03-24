#include "neuralNetwork.h"
#include <random>
#include <iostream>
#include <exception>
#include <algorithm>
#include <fstream>
#include <cstring>

neuralNetwork::neuralNetwork(std::vector<int> networkShape, float networkLearningRate, activationFunction networkActivation, errorFunction networkError, regularizationFunction networkRegularization, float regularizationLambda) {
    shape = networkShape;
    learningRate = networkLearningRate;
    activation = networkActivation;
    error = networkError;

    regularization = networkRegularization;
    lambda = regularizationLambda;

    nodesWithActivation.push_back(matrix(shape.at(0), 1));

    for (int layer = 1; layer < shape.size(); layer++) {
        nodes.push_back(matrix(shape.at(layer), 1));
        nodesWithActivation.push_back(matrix(shape.at(layer), 1));
        
        biases.push_back(matrix(shape.at(layer), 1));
        weights.push_back(matrix(shape.at(layer), shape.at(layer - 1)));
    }

    initParameters();
}

void neuralNetwork::initParameters() {
    for (int layer = 1; layer < shape.size(); layer++) {
        biases.at(layer - 1).applyFunction(getRandomNumber);

        weights.at(layer - 1).applyFunction(getRandomNumber);
    }
}
    
float neuralNetwork::getRandomNumber(float param) {
    std::random_device seedGenerator;
    std::mt19937 generator(seedGenerator());
    std::uniform_real_distribution<float> distribution(-1, 1);

    return distribution(generator);
}

float neuralNetwork::sigmoidF(const float x) {
    return 1.0f / (1 + std::exp(-x));
}

float neuralNetwork::sigmoidFPrime(const float x) {
    return (std::exp(-x)) / (std::pow(1 + std::exp(-x), 2)); 
}

float neuralNetwork::mseF(const float a, const float t, const int n) {
    return (1/(float)n) * std::pow(a - t, 2);
}

float neuralNetwork::mseFPrime(const float a, const float t, const int n) {
    return (1/(float)n) * 2 * (a - t);
}

float neuralNetwork::L2F(std::vector<matrix>& w, const int n, const float lambda) {
    float weightSum = 0; 

    for (int layer = 0; layer < w.size(); layer++) {
        for (int i = 0; i < w.at(layer).rows; i++) {
            for (int j = 0; j < w.at(layer).cols; j++) {
                weightSum += w.at(layer)(i, j) * w.at(layer)(i, j);
            }
        }
    }

    return (lambda/(2.0f * n)) * weightSum;
}

float neuralNetwork::L2FPrime(const float w, const int n, const float lambda) {
    return (lambda / n) * w;
}

matrix neuralNetwork::feedfoward(const matrix &input) {
    if (input.cols != 1 || input.rows != shape.at(0)) throw std::logic_error("Bad input dimensions");

    matrix layerMatrix = input;

    nodesWithActivation.at(0) = input;

    for (int layer = 1; layer < shape.size(); layer++) {
        layerMatrix = weights.at(layer - 1) * layerMatrix + biases.at(layer - 1);

        nodes.at(layer - 1) = layerMatrix;

        layerMatrix.applyFunction(activation.f);

        nodesWithActivation.at(layer) = layerMatrix;
    }

    return layerMatrix;
}

void neuralNetwork::train(std::vector<trainingExample> examples, int epochs, int miniBatchSize, bool shuffleData) {
    if (miniBatchSize > examples.size()) throw std::logic_error("Minibatch size can't be bigger than number of examples");

    std::random_device seedGenerator;
    std::mt19937 generator(seedGenerator());

    for (int epoch = 0; epoch < epochs; epoch++) { 
        std::cout << "Epoch " << epoch + 1 << " of " << epochs << " ; cost=" << getCostOverExamples(examples) << std::endl;

        if (shuffleData) std::shuffle(examples.begin(), examples.end(), generator); 

        for (int miniBatchIter = 0; miniBatchIter <= examples.size() - miniBatchSize; miniBatchIter += miniBatchSize) {
            std::vector<trainingExample> miniBatch(examples.begin() + miniBatchIter, examples.begin() + miniBatchIter + miniBatchSize);

            gradientDescent(miniBatch);
        }
    }
}

void neuralNetwork::gradientDescent(std::vector<trainingExample>& miniBatch) {
    std::vector<matrix> weightsGradients;
    std::vector<matrix> biasesGradients;

    for (int layer = 1; layer < shape.size(); layer++) {
        biasesGradients.push_back(matrix(shape.at(layer), 1));
        weightsGradients.push_back(matrix(shape.at(layer), shape.at(layer - 1)));
    }

    for (trainingExample& example : miniBatch) backpropagate(example, weightsGradients, biasesGradients);

    for (int layer = 1; layer < shape.size(); layer++) {
        weights.at(layer - 1) = weightsGradients.at(layer - 1) * (-learningRate * (1.0f/miniBatch.size())) + weights.at(layer - 1);
        biases.at(layer - 1) = biasesGradients.at(layer - 1) * (-learningRate * (1.0f/miniBatch.size())) + biases.at(layer - 1);
    }
}

void neuralNetwork::backpropagate(trainingExample& example, std::vector<matrix>& weightsGradients, std::vector<matrix>& biasesGradients) {
    if (example.target.rows != shape.at(shape.size() - 1)) throw std::logic_error("Example target must be same dimensions as last layer in network.");
    if (example.input.rows != shape.at(0)) throw std::logic_error("Example must be same dimensions as first layer in network.");

    matrix networkOutput = feedfoward(example.input);

    std::vector<matrix> activationGradients = getActivationGradients(networkOutput, example.target);

    for (int layer = 1; layer < shape.size(); layer++) {
        for (int i = 0; i < shape.at(layer); i++) {
            for (int j = 0; j < shape.at(layer - 1); j++) {
                weightsGradients.at(layer - 1)(i, j) += activationGradients.at(layer - 1)(i, 0) * 
                                                        activation.fPrime(nodes.at(layer - 1)(i, 0)) * 
                                                        nodesWithActivation.at(layer - 1)(j, 0)  
                                                        + regularization.fPrime(weights.at(layer - 1)(i, j), shape.at(shape.size() - 1), lambda);
            }

            biasesGradients.at(layer - 1)(i, 0) += activationGradients.at(layer - 1)(i, 0) * activation.fPrime(nodes.at(layer - 1)(i, 0));
        }
    }
}

std::vector<matrix> neuralNetwork::getActivationGradients(matrix& networkOutput, matrix& target) {
    std::vector<matrix> activationGradients;

    for (int layer = 1; layer < shape.size(); layer++) activationGradients.push_back(matrix(shape.at(layer), 1));

    for (int node = 0; node < shape.at(shape.size() - 1); node++) 
        activationGradients.at(shape.size() - 2)(node, 0) = error.fPrime(networkOutput(node, 0), target(node, 0), shape.at(shape.size() - 1)); 

    for (int layer = shape.size() - 2; layer > 0; layer--) {
        for (int node = 0; node < shape.at(layer); node++) {
            for (int k = 0; k < shape.at(layer + 1); k++) {
                activationGradients.at(layer - 1)(node, 0) += activationGradients.at(layer)(k, 0)
                                                            * activation.fPrime(nodes.at(layer)(k, 0)) 
                                                            * weights.at(layer)(k, node); 
            }
        }
    }

    return activationGradients;
}

void neuralNetwork::save(const char* fileName) {
    std::ofstream outFile;

    outFile.open(fileName, std::ofstream::binary);

    for (matrix& layerWeightMatrix : weights) 
        outFile.write((char*)layerWeightMatrix.getDataPointer(), 
                             layerWeightMatrix.rows * layerWeightMatrix.cols * sizeof(float));

    for (matrix& layerBiasesMatrix : biases)
        outFile.write((char*)layerBiasesMatrix.getDataPointer(),
                             layerBiasesMatrix.rows * layerBiasesMatrix.cols * sizeof(float));

    outFile.close();
}

void neuralNetwork::load(const char *fileName) {
    std::ifstream inFile;

    inFile.open(fileName, std::ifstream::binary);

    for (matrix& layerWeightMatrix : weights) {
        const unsigned int dataSize = layerWeightMatrix.rows * layerWeightMatrix.cols * sizeof(float);

        inFile.read((char*) layerWeightMatrix.getDataPointer(), dataSize);
    }

    for (matrix& layerBiasesMatrix : biases) {
        const unsigned int dataSize = layerBiasesMatrix.rows * layerBiasesMatrix.cols * sizeof(float);

        inFile.read((char*) layerBiasesMatrix.getDataPointer(), dataSize);
    }

    inFile.close();
}

int neuralNetwork::oneHotIndex(matrix& out) {
    float largestVal = -100000;
    int indexOfLargestVal = 0;

    for (int i = 0; i < out.rows; i++) {
        if (out(i, 0) > largestVal) {
            largestVal = out(i, 0);
            indexOfLargestVal = i;
        }
    }

    return indexOfLargestVal;
}

float neuralNetwork::getCostOverExamples(std::vector<trainingExample>& examples) {
    float cost = 0;
    
    for (trainingExample& example : examples) {
        matrix out = feedfoward(example.input);

        for (int i = 0; i < out.rows; i++) {
            cost += error.f(out(i, 0), example.target(i, 0), out.rows);
        }

        cost += regularization.f(weights, out.rows, lambda);
    }

    return cost;
}

float neuralNetwork::getAccuracyOverExamples(std::vector<trainingExample>& examples) {
    int assertCount = 0;
    for (trainingExample& example : examples) {
        matrix out = feedfoward(example.input);
        if (oneHotIndex(out) == oneHotIndex(example.target)) 
            assertCount++;
    }
    return ((float)assertCount / examples.size()) * 100;
}

void neuralNetwork::print() {
    std::cout << "Network shape: ";
    for (int layer = 0; layer < shape.size(); layer++) std::cout << shape.at(layer) << " ";
    std::cout << std::endl;

    std::cout << "Weight matrices: " << std::endl;
    for (int layer = 1; layer < shape.size(); layer++) {
        weights.at(layer - 1).print();
        std::cout << std::endl;
    }

    std::cout << "Bias matrices: " << std::endl;
    for (int layer = 1; layer < shape.size(); layer++) {
        biases.at(layer - 1).print();
        std::cout << std::endl;
    }
}
