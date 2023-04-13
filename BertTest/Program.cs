using BertTest;
using System;
using System.Text.Json;

Console.WriteLine("Initializing Model...");
var model = new Bert(
    Directory.GetCurrentDirectory() + "\\Assets\\Vocabulary\\vocab.txt",
    Directory.GetCurrentDirectory() + "\\Assets\\Model\\bertsquad-10.onnx"
    );

string text = "Jim is walking through the woods";

Console.WriteLine("Predicting...");
var (tokens, probability) = model.Predict(
    text, 
    "Who was him?");

Console.WriteLine(JsonSerializer.Serialize(new
{
    Probability = probability,
    Tokens = tokens
}));

Console.WriteLine("Done.");
Console.ReadLine();
