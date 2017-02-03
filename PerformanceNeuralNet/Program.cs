using System;
using System.Threading;
using PerformanceNeuralNet.Networks;

namespace PerformanceNeuralNet
{
	public class Program
	{
		public static void Main(string[] args)
		{
			var trainingData = Data.Load(@"PerformanceData\final_training_data.csv", ClassificationColumn.Last);
			trainingData.Normalize();

			var validationData = Data.Load(@"PerformanceData\final_test_data.csv", ClassificationColumn.Last);
			validationData.Normalize();

			//var validationData2 = Data.Load(@"PerformanceData\200_training_data_winnativems.csv", ClassificationColumn.Last);
			//validationData2.Normalize();

			var cancellationTokenSrc = new CancellationTokenSource();

			var tester = new NeuralNetworksTester(new INeuralNetwork[] {
				//new BackPropagationLearningNeuralNetwork(0.1, 0.1, 1.0, trainingData.InputCount, 40, trainingData.ClassCount),
				//new BackPropagationLearningNeuralNetwork(0.1, 0.1, 0.1, trainingData.InputCount, 100, 40, trainingData.ClassCount),
				//new BackPropagationLearningNeuralNetwork(0.4, 0.4, 0.1, trainingData.InputCount, 200, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 1.0, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 1.0, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 1.0, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 0.1, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 0.1, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.1, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 0.001, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 0.001, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.001, trainingData.InputCount, 100, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 1.0, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 1.0, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 1.0, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 0.1, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 0.1, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.1, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 0.05, trainingData.InputCount, 40, 10, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.1, 0.001, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.012, 0.001, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.001, trainingData.InputCount, 200, 40, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.01, trainingData.InputCount, 40, trainingData.ClassCount),
				new ResilientBackpropagationLearningNeuralNetwork(0.05, 0.05, trainingData.InputCount, 100, 50, trainingData.ClassCount),
				//new DeltaRuleLearningNeuralNetwork(0.1, 2.0, trainingData.InputCount, trainingData.ClassCount),
				//new EvolutionaryLearningNeuralNetwork(40, 0.01, trainingData.InputCount, trainingData.ClassCount),
				//new PerceptronLearningNeuralNetwork(0.1, 2.0, trainingData.InputCount, trainingData.ClassCount),
				//new ResilientBackpropagationLearningNeuralNetwork(0.0012, 0.001, trainingData.InputCount, 200, 40, trainingData.ClassCount),
			});

			var trainingTask = tester.Train(trainingData, 0.01, cancellationTokenSrc.Token);
			while (!trainingTask.IsCompleted && !IsEscapedPressed())
			{
				Thread.Sleep(50);
			}

			if (!trainingTask.IsCompleted)
			{
				Console.WriteLine("CANCELLED WAITING FOR LAST PROCESS!");
				cancellationTokenSrc.Cancel();
			}
			trainingTask.WaitIgnoreCanceled();

			tester.Save();

			Console.WriteLine();
			Console.WriteLine("Training Data");
			tester.Validate(trainingData, false);

			Console.WriteLine();
			Console.WriteLine("Traning Validation Data");
			tester.Validate(validationData, false);

			//Console.WriteLine();
			//Console.WriteLine("Win Validation Data");
			//tester.Validate(validationData2, true, @"C:\tmp");

			Console.WriteLine();
			Console.WriteLine("Press any key to quit...");
			Console.ReadKey(intercept: true);
		}

		private static bool IsEscapedPressed()
		{
			return Console.KeyAvailable
				&& Console.ReadKey(intercept: true).Key == ConsoleKey.Escape;
		}
	}
}
