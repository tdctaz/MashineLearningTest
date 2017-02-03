using System.Linq;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace PerformanceNeuralNet.Networks
{
	public class BackPropagationLearningNeuralNetwork : NeuralNetworkBase<ActivationNetwork>
	{
		private readonly int[] _layerCounts;
		private readonly double _sigmoidAlphaValue = 2.0;
		private readonly double _learningRate = 0.1;
		private readonly double _momentum = 0.1;

		private BackPropagationLearningNeuralNetwork(double learningRate, double momentum, double sigmoidAlphaValue)
		{
			_sigmoidAlphaValue = sigmoidAlphaValue;
			_learningRate = learningRate;
			_momentum = momentum;
		}

		public BackPropagationLearningNeuralNetwork(double learningRate, double momentum, double sigmoidAlphaValue, int inputCount, params int[] neuronCounts)
			: this(learningRate, momentum, sigmoidAlphaValue)
		{
			_layerCounts = new [] { inputCount }.Concat(neuronCounts).ToArray();
			var activationFunction = new BipolarSigmoidFunction(_sigmoidAlphaValue);
			_network = new ActivationNetwork(activationFunction, inputCount, neuronCounts);
			_network.Randomize();
		}

		public override string Name => $"{GetType().Name}-{string.Join("-", _layerCounts)}-{_learningRate}-{_momentum}-{_sigmoidAlphaValue}"
			.Replace(".", "_")
			.Replace(",", "_");

		protected override ISupervisedLearning CreateTeacher()
		{
			var teacher = new BackPropagationLearning(_network)
			{
				LearningRate = _learningRate,
				Momentum = _momentum
			};

			return teacher;
		}
	}
}
