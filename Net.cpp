#include "Net.h"
#include <cassert>

Net::Net(vector<unsigned> topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layers());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, 0));
		}
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//forward propagation
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layers &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer); // feed forward for neuron
		}
	}
}

void Net::backProp(const vector<double>& targetVals)
{

}