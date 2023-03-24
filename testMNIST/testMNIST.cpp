#include <iostream>
#include "readData.h"
#include "../neuralNetwork.h"
#define MAX_VALUE_OF_PIXEL 255

float normalizeValue(float value) {
    return value / MAX_VALUE_OF_PIXEL;
}

int main() {
    std::vector<trainingExample> trainingExamples = readMNISTData("mnist60KTrainingImages.bytes", "mnist60KTrainingLabels.bytes", 60000);
    std::vector<trainingExample> testingExamples = readMNISTData("mnist10KTestingImages.bytes", "mnist10KTestingLabels.bytes", 10000);
    
    for (trainingExample& t : trainingExamples) t.input.applyFunction(normalizeValue);

    neuralNetwork nn({784, 30, 10}, 0.5f, nn.sigmoid, nn.mse, nn.L2, 0.01f);

    //nn.load("mnistTrained.net");

    nn.train(trainingExamples, 5, 10, true);
    std::cout << "Accuracy over testing data: " << nn.getAccuracyOverExamples(testingExamples) << std::endl << std::endl;

    nn.save("mnistTrained.net");

    while (true) {
        int index;
        std::cin >> index;
        printExample(trainingExamples.at(index));

        matrix out = nn.feedfoward(trainingExamples.at(index).input);
        std::cout << "Prediction: " << neuralNetwork::oneHotIndex(out) << std::endl;
    }

    return 0;
}
