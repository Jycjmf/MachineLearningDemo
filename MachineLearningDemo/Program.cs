using Microsoft.ML;
using System;
using Microsoft.ML.Data;

namespace MachineLearningDemo
{
    public class Input
    {
        [LoadColumn(1)]
        public float Bathrooms;

        [LoadColumn(2)] 
        public float Bedrooms;

        [LoadColumn(3)]
        public float FinishedSquareFeet;

        [LoadColumn(5),ColumnName("Label")]//use ColumnName to specify which data you want to predict,Label is how ML.NET calls the 'target variable', the one that you want to predict, based on the other variables, called features.
        public float LastSoldPrice;

        [LoadColumn(9)]
        public float TotalRooms;

        [LoadColumn(10)]
        public string UseCode;
    }

    public class Output
    {
        [ColumnName("Score")]//Name of the tensor that will contain the output scores of the last layer when transfer learning is done. The default tensor name is "Score".
        public float Price;
    }
    internal class Program
    {
        private static readonly string Path = ".\\Data\\pacific-heights.csv";
        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
        //Load the data
            var data = context.Data.LoadFromTextFile<Input>(Path, hasHeader: true, separatorChar: ',');
            //Split the data
            var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;
            //var pipeline = context.Transforms.NormalizeMinMax("PovertyRate")
            //    .Append(context.Transforms.Concatenate("Features", "PovertyRate"))
            //    .Append(context.Regression.Trainers.Ols());
            //使用独热编码
            var pipeline = context.Transforms.Categorical
                .OneHotEncoding(inputColumnName: "UseCode", outputColumnName: "UseCodeEncoded")
                .Append(context.Transforms.Concatenate("Features", "UseCodeEncoded", "Bathrooms", "Bedrooms",
                    "TotalRooms", "FinishedSquareFeet")).Append(context.Regression.Trainers.FastForest(numberOfTrees:200,minimumExampleCountPerLeaf:4));

            var model = pipeline.Fit(trainData);// Train the model
            //Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = context.Regression.Evaluate(predictions);//metrics：指标

            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);// Create PredictionEngines
            var _input = new Input() { PovertyRate = 19.7f };
            var prediction = predictor.Predict(_input);
            Console.WriteLine($"Predict birth rate is:{prediction.BirthRate}");

        }
    }
}
