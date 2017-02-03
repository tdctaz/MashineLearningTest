using AForge.Neuro;
using AForge.Neuro.Learning;

namespace PerformanceNeuralNet.Networks
{
	public class EvolutionaryLearningNeuralNetwork : NeuralNetworkBase<ActivationNetwork>
	{
		private readonly int[] _layerCounts;
		private readonly int _populationSize;
		private readonly double _sigmoidAlphaValue;

		private EvolutionaryLearningNeuralNetwork(int populationSize, double sigmoidAlphaValue)
		{
			_populationSize = populationSize;
			_sigmoidAlphaValue = sigmoidAlphaValue;
		}

		public EvolutionaryLearningNeuralNetwork(int populationSize, double sigmoidAlphaValue, int inputCount, int neuronCounts)
			: this(populationSize, sigmoidAlphaValue)
		{
			_layerCounts = new[] { inputCount, neuronCounts };
			var activationFunction = new BipolarSigmoidFunction(_sigmoidAlphaValue);
			_network = new ActivationNetwork(activationFunction, inputCount, neuronCounts);
			_network.Randomize();
		}

		public override string Name => $"{GetType().Name}-{string.Join("-", _layerCounts)}-{_populationSize}-{_sigmoidAlphaValue}"
			.Replace(".", "_")
			.Replace(",", "_");

		protected override ISupervisedLearning CreateTeacher()
		{
			var teacher = new EvolutionaryLearning(_network, _populationSize);

			return teacher;
		}
	}
}
