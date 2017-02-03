using System.Linq;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace PerformanceNeuralNet.Networks
{
	public class ResilientBackpropagationLearningNeuralNetwork : NeuralNetworkBase<ActivationNetwork>
	{
		private readonly int[] _layerCounts;
		private readonly double _sigmoidAlphaValue;
		private readonly double _learningRate;

		private ResilientBackpropagationLearningNeuralNetwork(double learningRate, double sigmoidAlphaValue)
		{
			_sigmoidAlphaValue = sigmoidAlphaValue;
			_learningRate = learningRate;
		}

		public ResilientBackpropagationLearningNeuralNetwork(double learningRate, double sigmoidAlphaValue, int inputCount, params int[] neuronCounts)
			: this(learningRate, sigmoidAlphaValue)
		{
			_layerCounts = new[] {inputCount}.Concat(neuronCounts).ToArray();
			var activationFunction = new BipolarSigmoidFunction(_sigmoidAlphaValue);
			_network = new ActivationNetwork(activationFunction, inputCount, neuronCounts);
			_network.Randomize();
		}

		public override string Name => $"{GetType().Name}-{string.Join("-", _layerCounts)}-{_learningRate}-{_sigmoidAlphaValue}"
			.Replace(".", "_")
			.Replace(",", "_");

		protected override ISupervisedLearning CreateTeacher()
		{
			var teacher = new ResilientBackpropagationLearning(_network)
			{
				LearningRate = _learningRate,
			};

			return teacher;
		}
	}
}
