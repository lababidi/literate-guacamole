
#include <tiffio.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;
using namespace std;

int tiff_file(const char *file_name){
  TIFF *image;
  uint32 width, height;
  int c1, t1, imagesize;
  uint32 r1;
  int nsamples;
  unsigned char *scanline=NULL;

  uint16 BitsPerSample;           // normally 8 for grayscale image
  uint16 SamplesPerPixel;         // normally 1 for grayscale image
  uint16 i;

  // Open the TIFF image
  if((image = TIFFOpen(file_name, "r")) == NULL){
    fprintf(stderr, "Could not open incoming image\n");
    exit(42);
  }

  // Find the width and height of the image
  TIFFGetField(image, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(image, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
  TIFFGetField(image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
  imagesize = height * width + 1;    //get image size
  cout<<height<<" "<<width<<endl;

  //allocate memory for reading tif image
  scanline = (unsigned char *)_TIFFmalloc(SamplesPerPixel*width);
  if (scanline == NULL){
    fprintf (stderr,"Could not allocate memory!\n");
    exit(0);
  }

  fprintf(stderr,"W=%i H=%i BitsPerSample=%i SamplesPerPixel=%i\n", width, height,BitsPerSample,SamplesPerPixel);
  for (r1 = 0; r1 < height; r1++)
  {
    TIFFReadScanline(image, scanline, r1, 0);
    for (c1 = 0; c1 < width; c1++)
    {
      t1 = c1*SamplesPerPixel;

      for(i=0; i<SamplesPerPixel; i++)
        printf("%u \t", *(scanline + t1+i));
      printf("\n");
    }
  }

  _TIFFfree(scanline); //free allocate memory

  TIFFClose(image);
  return 1;
}

int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "/Users/mahmoud/t2/FCN.tensorflow/models/graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape({300, 300, 8}));
//  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
//    { "b", b },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  tiff_file("/Users/mahmoud/t2/FCN.tensorflow/powerplant/images/04c63991-75f8-4a9c-a071-8988934e21e9.tif");

  // Run the session, evaluating our "c" operation from the graph
  status = session->Run(inputs, {"inference/prediction"}, {}, &outputs);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}