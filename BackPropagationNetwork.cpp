#include "BackPropagationNetwork.h"

BackPropagationNetwork::BackPropagationNetwork(
        std::vector<int> layers, float learningSpeed) {

    this->_layers = layers;
    this->_learningSpeed = learningSpeed;

    this->InitAxons();
    this->InitNeurons();
    this->InitPrevValues();

    this->FillAxonsRand();

}

BackPropagationNetwork::~BackPropagationNetwork() {

    this->DeleteAxons();
    this->DeleteNeurons();
    this->DeletePrevValues();

}

float *BackPropagationNetwork::Learn(float *input, float *rightAnswers) {

    this->GroundNeurons();
    this->GroundPrevValues();

    this->AssignInput(input);
    this->Propagate();
    this->ChangeAxonsWithBP(rightAnswers);

    return this->_neurons[this->_layers.size() - 1];

}

float *BackPropagationNetwork::Propagate(float *input) {

    this->GroundNeurons();

    this->AssignInput(input);
    this->Propagate();

    return this->_neurons[this->_layers.size() - 1];

}

void BackPropagationNetwork::InitAxons() {

    this->_axons = new float**[this->_layers.size() - 1];

    for(int i = 0; i < this->_layers.size() - 1; i++) {

        this->_axons[i] = new float*[this->_layers[i]];

        for(int j = 0; j < this->_layers[i]; j++) {

            this->_axons[i][j] = new float[_layers[i + 1]];

        }

    }

}

void BackPropagationNetwork::DeleteAxons() {

    for(int i = 0; i < this->_layers.size() - 1; i++) {

        for(int j = 0; j < this->_layers[i]; j++) {

            delete[] this->_axons[i][j];

        }

        delete[] this->_axons[i];

    }

    delete[] this->_axons;

}

void BackPropagationNetwork::FillAxonsRand() {

    for(int i = 0; i < this->_layers.size() - 1; i++) {

        for(int j = 0; j < this->_layers[i]; j++) {

            for(int l = 0; l < this->_layers[i + 1]; l++) {

                this->_axons[i][j][l] = this->RandFloat();

            }

        }

    }

}

void BackPropagationNetwork::InitNeurons() {

    this->_neurons = new float*[this->_layers.size()];

    for(int i = 0; i < this->_layers.size(); i++) {

        this->_neurons[i] = new float[this->_layers[i]];

    }

}

void BackPropagationNetwork::DeleteNeurons() {

    for(int i = 0; i < this->_layers.size(); i++) {

        delete[] this->_neurons[i];

    }

    delete[] this->_neurons;

}

void BackPropagationNetwork::InitPrevValues() {

    this->_prevValues = new float*[this->_layers.size()];

    for(int i = 0; i < this->_layers.size(); i++) {

        this->_prevValues[i] = new float[this->_layers[i]];

    }

}

void BackPropagationNetwork::DeletePrevValues() {

    for(int i = 0; i < this->_layers.size(); i++) {

        delete[] this->_prevValues[i];

    }

    delete[] this->_prevValues;

}

void BackPropagationNetwork::AssignInput(float *input) {

    for(int i = 0; i < _layers[0]; i++) {

        this->_neurons[0][i] = input[i];

    }

}

void BackPropagationNetwork::Propagate() {

    for(int i = 0; i < this->_layers.size() - 1; i++) {

        this->ActivateAllNeuronsOnLayer(i);

        for(int j = 0; j < this->_layers[i]; j++) {

            for(int l = 0; l < this->_layers[i + 1]; l++) {

                this->_neurons[i + 1][l] +=
                        this->_neurons[i][j] * this->_axons[i][j][l];

            }

        }

    }

    this->ActivateAllNeuronsOnLayer(this->_layers.size() - 1);

}

void BackPropagationNetwork::ChangeAxonsWithBP(float *rightAnswers) {

    float** gradient = this->GetErrorGradient(rightAnswers);

    for(int i = 0; i < this->_layers.size() - 1; i++) {

        for(int j = 0; j < this->_layers[i]; j++) {

            for(int l = 0; l < this->_layers[i + 1]; l++) {

                this->_axons[i][j][l] -=
                        gradient[i + 1][l] *
                        this->DerivativeActivationFunction(
                                this->_prevValues[i + 1][l]) *
                        this->_neurons[i][j] * this->_learningSpeed;

            }

        }

    }

    for(int i = 0; i < this->_layers.size(); i++) {

        delete[] gradient[i];

    }

    delete[] gradient;

}

float **BackPropagationNetwork::GetErrorGradient(float *rightAnswers) {

    float** res = new float*[this->_layers.size()];

    res[this->_layers.size() - 1] = new float[
            this->_layers[this->_layers.size() - 1]];
    for(int i = 0; i < this->_layers[this->_layers.size() - 1]; i++) {

        res[_layers.size() - 1][i] =
                this->_neurons[this->_layers.size() - 1][i] -
                rightAnswers[i];

    }

    this->error = 0.0f;
    for(int i = 0; i < this->_layers[this->_layers.size() - 1]; i++) {

        error += res[this->_layers.size() - 1][i];

    } error /= this->_layers[this->_layers.size() - 1];

    for(int i = this->_layers.size() - 2; i >= 0; i--) {

        res[i] = new float[this->_layers[i]];

        for(int j = 0; j < this->_layers[i]; j++) {

            for(int l = 0; l < this->_layers[i + 1]; l++) {

                res[i][j] =
                        res[i + 1][l] * this->_axons[i][j][l];

            }

        }

    }

    return res;

}

void BackPropagationNetwork::GroundNeurons() {

    for(int i = 0; i < this->_layers.size(); i++) {

        for(int j = 0; j < this->_layers[i]; j++) {

            this->_neurons[i][j] = 0.0f;

        }

    }

}

void BackPropagationNetwork::GroundPrevValues() {

    for(int i = 0; i < this->_layers.size(); i++) {

        for(int j = 0; j < this->_layers[i]; j++) {

            this->_prevValues[i][j] = 0.0f;

        }

    }

}

float BackPropagationNetwork::ActivationFunction(float v) {

    return (1.0f / (1.0f + (float)(pow(E, (double)-v))));

}

float BackPropagationNetwork::DerivativeActivationFunction(float v) {

    float b = ActivationFunction(v);
    return (b * (1 - b));

}

void BackPropagationNetwork::ActivateAllNeuronsOnLayer(int layer) {

    for(int i = 0; i < this->_layers[layer]; i++) {

        float b = ActivationFunction(this->_neurons[layer][i]);
        this->_prevValues[layer][i] = this->_neurons[layer][i];
        this->_neurons[layer][i] = b;

    }

}

float BackPropagationNetwork::RandFloat() {

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(-1, 1);
    return distr(eng);

}


