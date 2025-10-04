import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'result_page.dart';

class CameraPage extends StatefulWidget {
  final String selectedModel;
  final String actualClassId; // renamed for consistency

  const CameraPage({
    super.key,
    required this.selectedModel,
    required this.actualClassId,
  });

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _controller;
  bool _isCameraReady = false;
  bool _isDetecting = false;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    var status = await Permission.camera.status;
    if (!status.isGranted) {
      status = await Permission.camera.request();
      if (!status.isGranted) return;
    }

    final cameras = await availableCameras();
    final camera = cameras.first;
    _controller = CameraController(camera, ResolutionPreset.low);
    await _controller!.initialize();

    if (!mounted) return;
    setState(() {
      _isCameraReady = true;
    });

    _startDetectionLoop();
  }

  void _startDetectionLoop() {
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!_isDetecting) {
        _isDetecting = true;
        _detectInsect();
      }
    });
  }

  Future<void> _detectInsect() async {
    if (!_controller!.value.isInitialized) return;

    final image = await _controller!.takePicture();
    final uri = Uri.parse("http://YOUR.IP.ADDRESS.HERE:8000/predict");

    final request = http.MultipartRequest('POST', uri);
    request.fields['model'] = widget.selectedModel;
    request.files.add(await http.MultipartFile.fromPath('image', image.path));
    final response = await request.send();

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      final decoded = json.decode(respStr);

      if (decoded.containsKey('class_id') && decoded['class_id'] != -1) {
        _timer?.cancel();
        _controller?.dispose();

        if (!mounted) return;

        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ResultPage(
              imagePath: image.path,
              predictedClassId: decoded['class_id'].toString(),
              predictedClassName: decoded['class_name'],
              actualClassId: widget.actualClassId,
              selectedModel: widget.selectedModel,
            ),
          ),
        );
      }
    } else {
      print("Server error: ${response.statusCode}");
      print(await response.stream.bytesToString());
    }

    _isDetecting = false;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Camera")),
      body: _isCameraReady
          ? CameraPreview(_controller!)
          : const Center(child: CircularProgressIndicator()),
    );
  }
}
