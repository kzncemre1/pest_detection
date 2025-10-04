import 'package:deneme_projesi/camera.dart';
import 'package:deneme_projesi/image.dart';
import 'package:flutter/material.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
     debugShowCheckedModeBanner: false,
      home:  MyHomePage(title: "Insect Detection")

    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage( {super.key, required this.title});

  final String title ;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
 
  String? _selectedModel  ;
  @override
  Widget build(BuildContext context) {
    
    return Scaffold(
      
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.green,
        title: Text(widget.title,style: TextStyle(fontWeight:FontWeight.bold,color: Colors.black) ,),
        
      ),
      body: Center( 

          child: Column(

            children: [
              SizedBox(height: 15,),
             Text(
                "Please select a model:",
                style: TextStyle(
                  fontFamily: 'Roboto', 
                  fontSize: 18.0, 
                  fontWeight: FontWeight.bold, 
                  color: Colors.blueGrey[800], 
                  letterSpacing: 0.5,
                  height: 1.2, 
                  shadows: [ 
                    Shadow(
                      offset: Offset(1.0, 1.0),
                      blurRadius: 1.0,
                      color: Color.fromARGB(100, 0, 0, 0), 
                    ),
                  ],
                 
                ),
                
                textAlign: TextAlign.center, 
                overflow: TextOverflow.ellipsis, 
                maxLines: 1, 
              ),
              SizedBox(height: 15,),
              RadioListTile<String>(
                title: const Text("ConvNeXt Tiny", style: TextStyle(fontWeight: FontWeight.bold),),
                value: "tiny",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!;
                  });
                },
              ),
              RadioListTile<String>(
                title: const Text("ConvNeXt Small", style: TextStyle(fontWeight: FontWeight.bold)),
                value: "small",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!;
                  });
                },
              ),
               RadioListTile<String>(
                title: const Text("ConvNeXt Base", style: TextStyle(fontWeight: FontWeight.bold)),
                value: "base",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!; // ! sign guarantees that value is not null, if it is null it will throw a runtime error
                  });
                },
              ),
              RadioListTile<String>(
                title: const Text("Swin Transformer", style: TextStyle(fontWeight: FontWeight.bold)),
                value: "swin",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!; 
                  });
                },
              ),
              RadioListTile<String>(
                title: const Text("Big Transfer", style: TextStyle(fontWeight: FontWeight.bold)),
                value: "bigT",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!; 
                  });
                },
              ),
              RadioListTile<String>(
                title: const Text("ConvMixer", style: TextStyle(fontWeight: FontWeight.bold)),
                value: "convm",
                groupValue: _selectedModel,
                onChanged: (value) {
                  setState(() {
                    _selectedModel = value!; 
                  });
                },
              ),
              ElevatedButton(
                onPressed: _selectedModel == null
                    ? null
                    : () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => NetworkImagePage(selectedModel: _selectedModel!),
                          ),
                        );
                      },
                child: const Text("Load the image", style: TextStyle(color: Colors.green),),
              ),


            ],
          )         
          
        
      )
      
      
    );
  }
}
