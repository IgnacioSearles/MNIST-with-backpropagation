#include "neuralNetwork.h"
#include <iostream>

int main() {
    neuralNetwork nn({2, 3, 3, 1}, 0.4f, nn.sigmoid, nn.mse, nn.L2, 0.0002f);

    std::vector<trainingExample> examples;
    matrix in1(2, 1), out1(1, 1); in1(0, 0) = 0; in1(1, 0) = 0; out1(0, 0) = 0; examples.push_back(trainingExample(in1, out1));
    matrix in2(2, 1), out2(1, 1); in2(0, 0) = 1; in2(1, 0) = 0; out2(0, 0) = 1; examples.push_back(trainingExample(in2, out2));
    matrix in3(2, 1), out3(1, 1); in3(0, 0) = 0; in3(1, 0) = 1; out3(0, 0) = 1; examples.push_back(trainingExample(in3, out3));
    matrix in4(2, 1), out4(1, 1); in4(0, 0) = 1; in4(1, 0) = 1; out4(0, 0) = 0; examples.push_back(trainingExample(in4, out4));

    nn.print();

    for (trainingExample& example : examples) {
        std::cout << "Input: " << example.input(0, 0) << ", " << example.input(1, 0) << "; Output:" << nn.feedfoward(example.input)(0, 0);
        std::cout << std::endl;
    }

    //nn.train(examples, 10000, 1, true);
    nn.load("trainedXOR.net");

    for (trainingExample& example : examples) {
        std::cout << "Input: " << example.input(0, 0) << ", " << example.input(1, 0) << "; Output:" << nn.feedfoward(example.input)(0, 0);
        std::cout << std::endl;
    }

    //nn.save("trainedXOR.net");
    nn.print();

    return 0;
}
