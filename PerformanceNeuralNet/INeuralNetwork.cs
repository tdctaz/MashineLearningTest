using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceNeuralNet
{
	public interface INeuralNetwork
	{
		string Name { get; }
		Task Train(Data trainingData, double learningErrorLimit);
		Task Train(Data trainingData, double learningErrorLimit, CancellationToken cancellationToken);
		void Load(string filename);
		void Save(string filename);
		Classification Classify(double[] input);
		IEnumerable<Classification> Classify(Data data);
	}
}
