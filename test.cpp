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


void sparseExpression(){
    int N = 2048;
    //out1 = input * W1
    //out2 = out1 * W2
    Format sparse({Dense, Sparse});

    Tensor<double> input({N, N}, sparse);

    Tensor<double> W1({N, N}, sparse);
    Tensor<double> W2({N, N}, sparse);

    Tensor<double> out1({N, N}, sparse);
    
    Tensor<double> out2({N, N}, sparse);

    fillTensor(&input, 256, 256);
    fillTensor(&W1, N, N);
    fillTensor(&W2, 256, 256);


    //input.pack();
    // W1.pack();
    // W2.pack();

    // Define tensor index variables
    IndexVar i("i"), j("j"), k("k");

    // out1 = input * W1
    out1(i, j) = sum(k, input(i, k) * W1(k, j));
    // out =  out1 * W2
    out2(i, j) = sum(k, out1(i, k) * W2(k, j));

    out1.pack();
    
    out1.compile();
    out1.assemble();
    out1.compute();

    out2.compile();
    out2.assemble();
    out2.compute();
}

void denseExpression(){
    int N = 2048;
    //out1 = input * W1
    //out2 = out1 * W2
    Format dm({Dense,Dense});

    Tensor<double> input({N, N}, dm);

    Tensor<double> W1({N, N}, dm);
    Tensor<double> W2({N, N}, dm);

    Tensor<double> out1({N, N}, dm);
    
    Tensor<double> out2({N, N}, dm);

    fillTensor(&input, 256, 256);
    fillTensor(&W1, N, N);
    fillTensor(&W2, 256, 256);


    //input.pack();
    // W1.pack();
    // W2.pack();

    // Define tensor index variables
    IndexVar i("i"), j("j"), k("k");

    // out1 = input * W1
    out1(i, j) = sum(k, input(i, k) * W1(k, j));
    // out =  out1 * W2
    out2(i, j) = sum(k, out1(i, k) * W2(k, j));

    out1.pack();
    
    out1.compile();
    out1.assemble();
    out1.compute();

    out2.compile();
    out2.assemble();
    out2.compute();
}


int main() {
    // denseExpression();
    sparseExpression();
    return 0;
}

