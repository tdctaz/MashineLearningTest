using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace PerformanceNeuralNet
{
	public abstract class NeuralNetworkBase<T> : INeuralNetwork
		where T : Network
	{
		protected T _network;

		public abstract string Name { get; }

		protected abstract ISupervisedLearning CreateTeacher();

		public Task Train(Data trainingData, double learningErrorLimit)
		{
			return Train(trainingData, learningErrorLimit, CancellationToken.None);
		}

		public Task Train(Data trainingData, double learningErrorLimit, CancellationToken cancellationToken)
		{
			var task = new Task(() => TrainingTask(trainingData, learningErrorLimit, cancellationToken), TaskCreationOptions.LongRunning);
			task.Start();
			return task;
		}

		private void TrainingTask(Data trainingData, double learningErrorLimit, CancellationToken cancellationToken)
		{
			var teacher = CreateTeacher();

			int iteration = 0;
			while (cancellationToken.IsCancellationRequested == false)
			{
				double error = teacher.RunEpoch(trainingData.Inputs, trainingData.Outputs) / trainingData.SampleCount;

				// check if we need to stop
				if (error <= learningErrorLimit)
				{
					Console.WriteLine($"DONE!!! ---- {Name} - {iteration}: error: {error}");
					break;
				}

				if (++iteration % 100 == 0)
				{
					Console.WriteLine($"{Name} - {iteration}: error: {error}");
				}
			}
		}

		public void Load(string filename)
		{
			_network = (T)Network.Load(filename);
		}

		public void Save(string filename)
		{
			_network.Save(filename);
		}

		public Classification Classify(double[] input)
		{
			return new Classification(_network.Compute(input));
		}

		public IEnumerable<Classification> Classify(Data data)
		{
			foreach (var input in data.Inputs)
			{
				yield return Classify(input);
			}
		}
	}
}
