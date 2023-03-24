#include "readData.h"
#include <fstream>
#include <iostream>

std::vector<trainingExample> readMNISTData(const char *imagesFilePath, const char *labelsFilePath, int numExamples) {
    std::vector<trainingExample> trainingExamples;

    std::ifstream imagesFile;
    imagesFile.open(imagesFilePath, std::ifstream::binary);

    std::ifstream labelsFile;
    labelsFile.open(labelsFilePath, std::ifstream::binary);

    imagesFile.seekg(IMAGES_FILE_METADATA_SIZE);
    labelsFile.seekg(LABELS_FILE_METADATA_SIZE);

    for (int i = 0; i < numExamples; i++) {
        unsigned char imageData[IMAGE_SIZE];
        imagesFile.read((char*) imageData, IMAGE_SIZE);

        matrix image(IMAGE_SIZE, 1);
        for (int j = 0; j < IMAGE_SIZE; j++) image(j, 0) = imageData[j];

        char labelData;
        labelsFile.read(&labelData, 1);

        matrix label(LABEL_SIZE, 1);
        label(labelData, 0) = 1;

        trainingExamples.push_back(trainingExample(image, label));
    }

    imagesFile.close();
    labelsFile.close();

    return trainingExamples;
}


void printExample(trainingExample& example) {
    std::cout << "Image of a " << neuralNetwork::oneHotIndex(example.target) << std::endl << std::endl;

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++)
            std::cout << " .*$#"[(int)(example.input(i * 28 + j, 0) * 255) / (int)51] << " ";

        std::cout << std::endl;
    }
}
