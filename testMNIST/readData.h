#pragma once
#include "../neuralNetwork.h"
#include <vector>

#define IMAGE_SIZE 28 * 28
#define LABEL_SIZE 10

#define IMAGES_FILE_METADATA_SIZE 4 * 4
#define LABELS_FILE_METADATA_SIZE 4 * 2

std::vector<trainingExample> readMNISTData(const char *imagesFilePath, const char *labelsFilePath, int numExamples);
void printExample(trainingExample& example);
