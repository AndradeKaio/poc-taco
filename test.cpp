#include <random>
#include <taco.h>
#include <iostream>

using namespace taco;


void fillTensorRandomly(Tensor<double> *t, int n, int k){
  std::random_device rd;
  std::mt19937 gen(rd());
  int min = 0, max = 1;
  std::uniform_int_distribution<int> dist(min, max);

  for(int i = 0; i < n; i++)
    for(int j = 0; j<k; j++){
        int b = dist(gen);
        if(b)
          t->insert({i,j}, 1.0);
        else
          t->insert({i,j}, 0.0);
    }
}

void fillTensor(Tensor<double> *t, int n, int k){
  std::random_device rd;
  std::mt19937 gen(rd());
  double min = 1.0, max = 5.0;
  std::uniform_real_distribution<double> dist(min, max);

  for(int i = 0; i < n; i++)
    for(int j = 0; j<k; j++)
        t->insert({i,j}, dist(gen));
}

int main() {
    // Define tensor dimensions
    int N = 2048;
    //out1 = input * W1
    //out2 = out1 * W2
    Tensor<double> input({N, N}, Format({Sparse, Dense}));

    Tensor<double> W1({N, N}, Format({Dense, Dense}));
    Tensor<double> W2({N, N}, Format({Dense, Dense}));

    Tensor<double> dense({N, N}, Format({Dense, Dense}));
    Tensor<double> csr({N, N}, Format({Sparse, Dense}));

    Tensor<double> out({N, N}, Format({Dense, Dense}));
    
    Tensor<double> out1, out2;

    fillTensor(&input, 25, 25);
    fillTensor(&W1, N, N);
    fillTensor(&W2, 25, 25);


    input1.pack();
    // W1.pack();
    // W2.pack();

    // Define tensor index variables
    IndexVar i("i"), j("j"), k("k");

    out1 = csr;
    // out1 = input * W1
    out1(i, j) = sum(k, input(i, k) * W1(k, j));
    // out =  out1 * W2
    out(i, j) = sum(k, out1(i, k) * W2(k, j));

    out1.pack();
    
    out1.compile();
    out1.assemble();
    out1.compute();

    out.compile();
    out.assemble();
    out.compute();

    // Print result
    // std::cout << out << std::endl;

    return 0;
}

