#ifndef XCRYPT_GEN1_BACKPROPAGATIONNETWORK_H
#define XCRYPT_GEN1_BACKPROPAGATIONNETWORK_H

#define E 2.71828

#include <vector>
#include <math.h>
#include <random>


class BackPropagationNetwork {
public:

    float error;

    BackPropagationNetwork(
            std::vector<int> layers,
            float learningSpeed);
    ~BackPropagationNetwork();

    float* Learn(
            float* input, float* rightAnswers);
    float* Propagate(
            float* input);

private:

    std::vector<int> _layers;
    float*** _axons;
    float** _neurons;
    float** _prevValues;
    float _learningSpeed;

    void InitAxons();
    void DeleteAxons();
    void FillAxonsRand();

    void InitNeurons();
    void DeleteNeurons();

    void InitPrevValues();
    void DeletePrevValues();

    void AssignInput(float* input);
    void Propagate();
    void ChangeAxonsWithBP(float* rightAnswers);
    float** GetErrorGradient(float* rightAnswers);

    void GroundNeurons();
    void GroundPrevValues();

    float ActivationFunction(float v);
    float DerivativeActivationFunction(float v);
    void ActivateAllNeuronsOnLayer(int layer);

    float RandFloat();

};


#endif //XCRYPT_GEN1_BACKPROPAGATIONNETWORK_H
