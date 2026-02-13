#include "Neuron.h"

#include <cstdlib>

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		pair<double, double> weightDeltaWeight = { rand() / (double)RAND_MAX, 0.0 };
		m_outputWeights.push_back(weightDeltaWeight);
	}

	m_myIndex = myIndex;
}


void Neuron::feedForward(const Layers &prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].first;
	}

	m_outputVal = Neuron::activationFunction(sum);
}