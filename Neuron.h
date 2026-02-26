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
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layers &nextLayer);
	void updateInputWeights(Layers &prevLayer);

private:
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	static double activationFunction(double x) { return tanh(x); }
	static double activationFunctionDerivative(double x) { return 1.0 - x * x; } //approx
	double sumDOW(const Layers &nextLayer) const;
	double m_outputVal;
	vector<pair<double,double>> m_outputWeights; // weight, deltaWeight
	unsigned m_myIndex;
	double m_gradient;
};
