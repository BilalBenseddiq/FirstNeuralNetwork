////https://vimeo.com/19569529 (par David Miller)

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;



// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
	if (!m_trainingDataFile.is_open()) {
		cout << " Failed to open train data file" << endl;
	}
	else {
		cout << "Train Data File Opened OK" << endl;
	}
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
	inputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	string line;
	getline(m_trainingDataFile, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}



struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ********************** Class Neuron *******************
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) {	m_outputVal = val;	}
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..1.0] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex; //the number of the neuron inside the current layer
	double m_gradient;
	double sumDOW(const Layer &nextLayer) const;
};

double Neuron::eta = 0.15; // [0.0..1.0] overall net training rate
double Neuron::alpha = 0.5; // [0.0..1.0] multiplier of last weight change [momentum]

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer & prevLayer)
{
	double sum = 0.0;
	// Sum the previous layer's outputs (multiplied by the weight of the incoming connection), which are our inputs
	// Includes the bias node from the previous layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
		//cout << "from n: " << n << " val: " << prevLayer[n].getOutputVal() << " weight: " << prevLayer[n].m_outputWeights[m_myIndex].weight << endl;
		//cout << "sum: " << sum << endl;
		//cout << "index: " << m_myIndex << endl;
		m_outputVal = Neuron::transferFunction(sum);
	}
}

void Neuron::calcOutputGradients(double targetVal) 
{
	//there are other ways to calculate this gradients,
	//but this one keeps the training heading in a direction for it reduces the overall net error
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);

}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	//similar to the calculation of the output gradients, but since we don't have
	//target values, we use the sum of the derivatives of the weights of the next layer
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer & prevLayer)
{
	// The weights to be updated are in the Connection container 
	// in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			// Individual input magnified by the gradient and train rate (eta)
			// eta = overall net learning rate ( 0.0 - slow learner, 0.2 - medium learner, 1.0 - reckless learner)
			eta
			* neuron.getOutputVal()
			* m_gradient
			// Also add momentum (alpha) = a fraction of the previous delta weight (to avoid valleys)
			// alpha = momentum (0.0 - no momentum, 0.5 - moderate momentum)
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}

}

double Neuron::transferFunction(double x)
{
	//tanh - output range [-1.0..1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

double Neuron::sumDOW(const Layer & nextLayer) const
{
	double sum = 0.0;
	// Sum our contributions of the errors at the nodes we feed
	// (size -1 because we do not include the bias neuron)
	for (unsigned n = 0; n < nextLayer.size() -1; ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	
	return sum;
}

// ********************** Class Net **********************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 0.0; // Number of training samples to average over (default: 100)

void Net::feedForward(const vector<double> &inputVals) {

	//cout << "inputVals: " << inputVals.size() << " layer_0 size: " << m_layers[0].size() <<endl;

	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Forward Propagation
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		//number of neurons in that layer
		unsigned nbNeurons = m_layers[layerNum].size();
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned numNeuron = 0; numNeuron < nbNeurons - 1; ++numNeuron) {
			//cout << "prop layer: " << layerNum << " neur num: " << numNeuron << endl;
			m_layers[layerNum][numNeuron].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	// Calculate overall network error (erreur quadradtique moyenne)

	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() -1; ++n) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; //get average error squared
	m_error = sqrt(m_error); //erreur quadratique moyenne

	// Implement a recent average measurement

	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients

	for (unsigned n = 0; n < outputLayer.size() -1; ++n) {
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	// Calculate hidden layers gradients

	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	// For all layers from output to first hidden layer,
	// update connection weights.

	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() -1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(vector<double>& resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}


Net::Net(const vector<unsigned> &topology)
{
	// topology: m_layers = a vector of layers, where each Layer is a vector of Neurons
	//           ex: m_layers[0][3] = fourth neuron of the first layer
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutPuts = layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

		// We have made a new Layer, now fill it with neurons and
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutPuts, neuronNum));
			//cout << "Made a neuron!" << endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above.
		m_layers.back().back().setOutputVal(1.0);

	}

}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		cout << v[i] << " ";
	}

	cout << endl;
}

int main() {
	/********************************************************************
	* 	How to use:														*
	*																	*
	*	1.Prepare training data											*
	*	2.Send each of inputs to feed forward()							*
	*	3.For training, send each of desired outputs to backProp()		*
	*	4.Get the net's actual results back with getResults()			*
	*																	*
	*********************************************************************/

	TrainingData trainData("D:/Dev/Neural Networks/trainingData.txt");

	//e.g., {3,2,1}
	vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	// Main loop

	while (!trainData.isEof()) {
		++trainingPass;
		cout << endl << "Pass " << trainingPass;

		// Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		// Collect the net's actual output results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		// Train the net what the outputs should have been:
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		// Report how well the training is working, average over recent samples:
		cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;
	}

	cout << endl << "Done" << endl;
	

	return 0;
}


