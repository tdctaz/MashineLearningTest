using System;
using System.Linq;
using System.Threading.Tasks;

namespace PerformanceNeuralNet
{
	public static class TaskExtensions
	{
		public static void WaitIgnoreCanceled(this Task task)
		{
			try
			{
				task.Wait();
			}
			catch (AggregateException ae)
				when (ae.InnerExceptions.All(x => x is TaskCanceledException))
			{
				// Ignore
			}
			catch (TaskCanceledException)
			{
				// Ignore
			}
		}
	}
}
