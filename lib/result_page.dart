import 'dart:io';
import 'package:flutter/material.dart';

class ResultPage extends StatelessWidget {
  final String imagePath;
  final String predictedClassId;
  final String predictedClassName;
  final String actualClassId;
  final String selectedModel;

  const ResultPage({
    super.key,
    required this.imagePath,
    required this.predictedClassId,
    required this.predictedClassName,
    required this.actualClassId,
    required this.selectedModel,
  });

  @override
  Widget build(BuildContext context) {
    bool isCorrect = predictedClassId == actualClassId;

    return Scaffold(
      backgroundColor: Colors.green,
      appBar: AppBar(
        backgroundColor: Colors.lime,
        title: const Text("Prediction Result"),
        centerTitle: true,
      ),
      body: Center(
    
        child: SingleChildScrollView(
      
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.file(
                  File(imagePath),
                  height: 250,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 24),
              Text(
                "Predicted Class:",
                style: TextStyle(fontSize: 16, color: Colors.grey[600]),
              ),
              Text(
                "$predictedClassName (ID: $predictedClassId)",
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Text(
                "Actual Class:",
                style: TextStyle(fontSize: 16, color: Colors.grey[600]),
              ),
              Text(
                "ID: $actualClassId",
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 24),
              Text(
                isCorrect ? "✅ Correct Prediction" : "❌ Wrong Prediction",
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: isCorrect ? Colors.green : Colors.red,
                ),
              ),
              const SizedBox(height: 32),
              Text(
                "Model Used: $selectedModel",
                style: TextStyle(fontSize: 16, color: Colors.grey[700]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
