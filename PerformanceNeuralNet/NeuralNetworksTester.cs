using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PerformanceNeuralNet
{
	public class NeuralNetworksTester
	{
		private readonly INeuralNetwork[] _neuralNetworks;

		public NeuralNetworksTester(params INeuralNetwork[] neuralNetworks)
		{
			_neuralNetworks = neuralNetworks;
		}

		public async Task Train(Data trainingData, double learningErrorLimit, CancellationToken cancellationToken)
		{
			foreach (var neuralNetwork in _neuralNetworks)
			{
				TryLoad(neuralNetwork);
			}

			var traningTasks = _neuralNetworks.Select(x => x.Train(trainingData, learningErrorLimit, cancellationToken));
			await Task.WhenAll(traningTasks).ConfigureAwait(false);
		}

		public void Save()
		{
			foreach (var neuralNetwork in _neuralNetworks)
			{
				Save(neuralNetwork);
			}
		}

		private void TryLoad(INeuralNetwork network)
		{
			var directory = new DirectoryInfo(Directory.GetCurrentDirectory());
			var file = directory.EnumerateFiles($"{network.Name}-*.nn").OrderByDescending(x => x.Name).FirstOrDefault();
			if (file != null)
			{
				network.Load(file.FullName);
			}
		}

		private void Save(INeuralNetwork network)
		{
			var timestamp = DateTime.UtcNow.ToString("u").Replace("-", "").Replace(":", "").Replace("T", "");
			network.Save($"{network.Name}-{timestamp}.nn");
		}

		public void Validate(Data validationData, bool verbose, string outputPath = null)
		{
			foreach (var network in _neuralNetworks)
			{
				StringBuilder outputData = new StringBuilder();
				int successCount = 0;
				for (int i = 0; i < validationData.SampleCount; i++)
				{
					var results = network.Classify(validationData.Inputs[i]);

					CreateCsvRow(validationData, i, results, outputData);

					var error = CalcError(validationData.Outputs[i], results.Classifications);
					if (Success(validationData.Outputs[i], results.Classifications))
					{
						successCount++;
						if (verbose)
						{
							Console.WriteLine($"{network.Name} {i} SUCCESS: error {error}");
						}
					}
					else
					{
						if (verbose)
						{
							Console.WriteLine($"{network.Name} {i} FAILURE? error {error}");
							Console.WriteLine($"\t\t{string.Join("    ", results.Classifications.Select(x => $"{x:0.000}"))}");
							Console.WriteLine($"\t\t{string.Join("    ", validationData.Outputs[i].Select(x => $"{x:0.000}"))}");
						}
					}
				}

				if (verbose)
				{
					Console.WriteLine();
				}
				var successRate = (successCount / (double)validationData.SampleCount) * 100.0;
				Console.WriteLine($"{network.Name}: {successRate:0.00}%");

				if (outputPath != null)
				{
					File.WriteAllText($"{outputPath}\\results-{successRate:0.00}-{network.Name}.cvs", outputData.ToString());
				}
			}
		}

		private static void CreateCsvRow(Data validationData, int i, Classification results, StringBuilder outputData)
		{
			var inputs = string.Join(",", validationData.Inputs[i].Select(x => x.ToString(CultureInfo.InvariantCulture)));
			var output = validationData.Outputs[i].IndexOf(x => validationData.Outputs[i].Max() == x);
			var predictions = string.Join(",", results.Classifications.Select(x => x.ToString(CultureInfo.InvariantCulture)));
			outputData.AppendLine($"{inputs},{output},{predictions}");
		}

		private static bool Success(double[] expected, double[] actual)
		{
			double best = actual.Max();
			int bestIndex = -1;
			for (int i = 0; i < actual.Length; i++)
			{
				if (actual[i] == best)
					bestIndex = i;
			}

			return expected[bestIndex] > 0;
		}

		private static double CalcError(double[] expected, double[] actual)
		{
			double error = 0.0;
			for (int i = 0; i < expected.Length; i++)
			{
				error += Math.Abs(expected[i] - actual[i]);
			}
			return error;
		}
	}
}
