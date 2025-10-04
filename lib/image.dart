import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:deneme_projesi/camera.dart';

class NetworkImagePage extends StatefulWidget {
  final String selectedModel;
  const NetworkImagePage({super.key, required this.selectedModel});

  @override
  State<NetworkImagePage> createState() => _NetworkImagePageState();
}

class _NetworkImagePageState extends State<NetworkImagePage> {
  final String apiInfoUrl = "http://YOUR.IP.ADDRESS.HERE:8000/random-image-info";
  final String imageBaseUrl = "http://YOUR.IP.ADDRESS.HERE:8000/image/";

  String? imagePath;
  String? className;
  String? classId; // <- class_id burada tutuluyor
  bool isLoading = true;
  String? error;

  @override
  void initState() {
    super.initState();
    fetchRandomImage();
  }

  Future<void> fetchRandomImage() async {
    setState(() {
      isLoading = true;
      error = null;
    });

    try {
      final response = await http.get(Uri.parse(apiInfoUrl));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          imagePath = data["image_path"];
          classId = data["class_id"]; 
          className =
              "ID: ${data['class_id']} NAME: ${data['class_name']} (${data["file_name"]})";
          isLoading = false;
        });
      } else {
        setState(() {
          error = "Image not found";
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        error = "An error occured: $e";
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Randome Image")),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : error != null
              ? Center(child: Text(error!))
              : Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Expanded(
                      child: Center(
                        child: Image.network(
                          imageBaseUrl + imagePath!,
                          errorBuilder: (context, error, stackTrace) {
                            return const Text("Image could not be loaded");
                          },
                        ),
                      ),
                    ),
                    Text("$className", style: const TextStyle(fontSize: 18)),
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          ElevatedButton(
                            onPressed: fetchRandomImage,
                            child: const Text("New Image", style: TextStyle(color: Colors.green),),
                          ),
                          ElevatedButton(
                            onPressed: classId == null
                                ? null //Button should not be active without classId
                                : () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => CameraPage(
                                          selectedModel: widget.selectedModel,
                                          actualClassId: classId!,
                                        ),
                                      ),
                                    );
                                  },
                            child: const Text("Open the Camera", style: TextStyle(color: Colors.green),),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
    );
  }
}
