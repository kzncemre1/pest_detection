import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'dart:convert';

class MetricsPage extends StatefulWidget {
  
  final String selectedModel;
  const MetricsPage({Key? key, required this.selectedModel}) : super(key: key);
  
  @override
  _MetricsPageState createState() => _MetricsPageState();
}

class _MetricsPageState extends State<MetricsPage> {
  Map<String, dynamic>? metrics;
  bool isLoading = true;
  String? error;

  @override
  void initState() {
    super.initState();
    fetchMetrics();
  }

  Future<void> fetchMetrics() async {
    try {
      
      final uri = Uri.parse('http://YOUR.IP.ADDRESS.HERE:8000/metrics?model=${widget.selectedModel}');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        setState(() {
          metrics = jsonDecode(response.body);
          isLoading = false;
        });
      } else {
        throw Exception('Failed to retrieve metrics from server');
      }
    } catch (e) {
      setState(() {
        error = e.toString();
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Model Metrics')),
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : error != null
              ? Center(child: Text('Error: $error'))
              : SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Classification Report',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      SizedBox(height: 8),
                      _buildClassificationReport(metrics!['classification_report']),
                      SizedBox(height: 24),
                      Text('Confusion Matrix',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      SizedBox(height: 8),
                      _buildConfusionMatrix(metrics!['confusion_matrix']),
                    ],
                  ),
                ),
    );
  }

  Widget _buildClassificationReport(Map<String, dynamic> report) {
    final rows = <TableRow>[
      TableRow(
        decoration: BoxDecoration(color: Colors.grey[300]),
        children: ['Label', 'Precision', 'Recall', 'F1-Score', 'Support']
            .map((e) => Padding(
                  padding: EdgeInsets.all(8.0),
                  child: Text(e, style: TextStyle(fontWeight: FontWeight.bold)),
                ))
            .toList(),
      ),
    ];

    report.forEach((label, metrics) {
      if (metrics is Map) {
        rows.add(
          TableRow(
            children: [
              Text(label),
              Text(metrics['precision'].toStringAsFixed(2)),
              Text(metrics['recall'].toStringAsFixed(2)),
              Text(metrics['f1-score'].toStringAsFixed(2)),
              Text(metrics['support'].toString()),
            ].map((e) => Padding(padding: EdgeInsets.all(8.0), child: e)).toList(),
          ),
        );
      } else {
        // accuracy gibi tek değerler
        rows.add(
          TableRow(
            children: [
              Padding(padding: EdgeInsets.all(8.0), child: Text(label)),
              Padding(padding: EdgeInsets.all(8.0), child: Text(metrics.toString())),
              SizedBox(),
              SizedBox(),
              SizedBox(),
            ],
          ),
        );
      }
    });

    return Table(border: TableBorder.all(), children: rows);
  }

  Widget _buildConfusionMatrix(List<dynamic> matrix) {
    return Table(
      border: TableBorder.all(),
      children: matrix.map<TableRow>((row) {
        return TableRow(
          children: row.map<Widget>((cell) {
            return Padding(
              padding: EdgeInsets.all(8.0),
              child: Center(child: Text(cell.toString())),
            );
          }).toList(),
        );
      }).toList(),
    );
  }
}
