#include "Net.h"

int main()
{
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);

	Net myNet(topology);

	vector<double> inputVals;
	vector<double> targetVals;
	vector<double> resultVals;

	//myNet.feedForward(inputVals);
	//myNet.backProp(targetVals);
	//myNet.getResults(resultVals);
}