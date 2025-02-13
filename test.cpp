#include <vector>
#include <chrono>
#include <random>
#include <taco.h>
#include <iostream>


using namespace taco;
using namespace std;

using Matrix = vector<vector<double> >;

void fillTacoTensorRandomly(Tensor<double> *t, int N, double sparsity){
  int min = 0, max = 1;
  srand(time(0));

  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++){
      if ((rand() / (double)RAND_MAX) > sparsity)
          t->insert({i, j}, ((rand() % 10) + 1) * 1.0);
      else
          t->insert({i, j}, 0.0);
  }
}

void fillTacoTensor(Tensor<double> *t, int n, int k){
  std::random_device rd;
  std::mt19937 gen(rd());
  double min = 1.0, max = 5.0;
  std::uniform_real_distribution<double> dist(min, max);

  for(int i = 0; i < n; i++)
    for(int j = 0; j<k; j++)
        t->insert({i,j}, dist(gen));
}

void fillTensor(Matrix &t, int n, int k){
  std::random_device rd;
  std::mt19937 gen(rd());
  double min = 1.0, max = 5.0;
  std::uniform_real_distribution<double> dist(min, max);

  for(int i = 0; i < n; i++)
    for(int j = 0; j < k; j++)
      t[i][j] = dist(gen);
}

Matrix generateSparseMatrix(int rows, int cols, double sparsity) {
    Matrix matrix(rows, vector<double>(cols, 0.0));
    srand(time(0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if ((rand() / (double)RAND_MAX) > sparsity)
                matrix[i][j] = (rand() % 10) + 1;
        }
    }
    return matrix;
}

Matrix multiply(const Matrix &A, const Matrix &B, int N) {
    Matrix C(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

void printMatrix(const Matrix &M, int N) {
    cout << fixed << setprecision(2); // Set decimal precision to 2 places
    for (int i = 0; i < N; i++) {
        cout << "| ";
        for (int j = 0; j < N; j++)
            cout << setw(8) << M[i][j] << " "; // Align elements
        cout << " |" << endl;
    }
    cout << endl;
}

void denseExpression(){
    int N = 2048;
    int S = 256;

    Matrix input(N, vector<double>(N));
    Matrix W1(N, vector<double>(N));
    Matrix W2(N, vector<double>(N));
    
    fillTensor(input, S, S);
    fillTensor(W1, S, S);
    fillTensor(W2, S, S);
    
    auto start = std::chrono::high_resolution_clock::now();
    Matrix out1 = multiply(input, W1, N);
    Matrix out2 = multiply(out1, W2, N);
    //printMatrix(input, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
}

void tacoSparseExpression(bool random){
    int N = 2048;
    int S = 256;
    //out1 = input * W1
    //out2 = out1 * W2
    Format sparse({Sparse, Dense});

    Tensor<double> input({N, N}, sparse);

    Tensor<double> W1({N, N}, sparse);
    Tensor<double> W2({N, N}, sparse);

    Tensor<double> out1({N, N}, sparse);
    
    Tensor<double> out2({N, N}, sparse);

    if(!random){
      fillTacoTensor(&input, 256, 256);
      fillTacoTensor(&W1, N, N);
      fillTacoTensor(&W2, 256, 256);
    } else {
      fillTacoTensorRandomly(&input, N, 0.9);
      fillTacoTensorRandomly(&W1, N, 0.9);
      fillTacoTensorRandomly(&W2, N, 0.9);
    }


    //input.pack();
    // W1.pack();
    // W2.pack();

    // Define tensor index variables
    IndexVar i("i"), j("j"), k("k");

    // out1 = input * W1
    out1(i, j) = sum(k, input(i, k) * W1(k, j));
    // out =  out1 * W2
    out2(i, j) = sum(k, out1(i, k) * W2(k, j));

    auto start = std::chrono::high_resolution_clock::now();
    out1.pack();
    
    out1.compile();
    out1.assemble();
    out1.compute();

    out2.compile();
    out2.assemble();
    out2.compute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
}

void tacoDenseExpression(bool random){
    int N = 2048;
    //out1 = input * W1
    //out2 = out1 * W2
    Format dm({Dense,Dense});

    Tensor<double> input({N, N}, dm);

    Tensor<double> W1({N, N}, dm);
    Tensor<double> W2({N, N}, dm);

    Tensor<double> out1({N, N}, dm);
    
    Tensor<double> out2({N, N}, dm);

    if(!random){
      fillTacoTensor(&input, 256, 256);
      fillTacoTensor(&W1, N, N);
      fillTacoTensor(&W2, 256, 256);
    } else {
      fillTacoTensorRandomly(&input, N, 0.8);
      fillTacoTensorRandomly(&W1, N, 0.8);
      fillTacoTensorRandomly(&W2, N, 0.8);
    }

    //input.pack();
    // W1.pack();
    // W2.pack();

    // Define tensor index variables
    IndexVar i("i"), j("j"), k("k");

    // out1 = input * W1
    out1(i, j) = sum(k, input(i, k) * W1(k, j));
    // out =  out1 * W2
    out2(i, j) = sum(k, out1(i, k) * W2(k, j));

    auto start = std::chrono::high_resolution_clock::now();
    out1.pack();
    
    out1.compile();
    out1.assemble();
    out1.compute();

    out2.compile();
    out2.assemble();
    out2.compute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
}


int main() {
    // denseExpression();
    //tacoDenseExpression();
    tacoSparseExpression(true);
    return 0;
}

