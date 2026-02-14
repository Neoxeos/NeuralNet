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

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layers &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layers &nextLayer) const
{
	double sum = 0.0;

	// sum contributions of the weights to the errors in the next layer
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].first * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layers &prevLayer)
{
	// the weights to be updated are in the Connection container in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].second;

		double newDeltaWeight =
			// individual input, magnified by the gradient and train rate
			neuron.getOutputVal() * m_gradient * Net::getEta()
			// also add momentum = a fraction of the previous delta weight
			+ oldDeltaWeight * Net::getAlpha();

		neuron.m_outputWeights[m_myIndex].second = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].first += newDeltaWeight;
	}
}