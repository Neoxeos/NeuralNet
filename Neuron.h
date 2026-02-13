#pragma once

#include <vector>
#include <cmath>

using namespace std;

class Neuron;

typedef vector<Neuron> Layers;

using namespace std;

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(const Layers &prevLayer);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal() const { return m_outputVal; }

private:
	static double activationFunction(double x) { return tanh(x); }
	static double activationFunctionDerivative(double x) { return 1.0 - x * x; } //approx
	double m_outputVal;
	vector<pair<double,double>> m_outputWeights; // weight, deltaWeight
	unsigned m_myIndex;
};
