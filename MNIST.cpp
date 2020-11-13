#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
const int N = 1e5 + 7;
const double PI = acos(-1.0);


class Numerical {
    public:
        void randomNormal(vector<vector<double> > &vec, int n, int m, double std) {
            vec.resize(n, vector<double>(m));
            for(int i = 0; i < n; i ++) {
                for(int j = 0; j < m; j ++) {
                    double u1 = rand() * 1.0 / RAND_MAX, u2 = rand() * 1.0 / RAND_MAX;
                    double x = cos(2 * PI * u1) * sqrt(-2.0 * (log(u2) / log(exp(1.0))));
                    vec[i][j] = std * x;
                }
            }
        }
        vector<vector<double> > dot(vector<vector<double> > v1, vector<vector<double> > v2) {
            vector<vector<double> > res(v1.size(), vector<double>(v2[0].size(), 0.0));
            for(int i = 0; i < v1.size(); i ++) {
                for(int k = 0; k < v2.size(); k ++) {
                    for(int j = 0; j < v2[0].size(); j ++) {
                        res[i][j] += v1[i][k] * v2[k][j];
                    }
                }
            }
            return res;
        }
        vector<vector<double> > subtract(vector<vector<double> > v1, vector<vector<double> > v2) {
            vector<vector<double> > res(v1.size(), vector<double>(v1[0].size()));
            for(int i = 0; i < v1.size(); i ++) {
                for(int j = 0; j < v1[i].size(); j ++) {
                    res[i][j] = v1[i][j] - v2[i][j];
                }
            }
            return res;
        }
        vector<vector<double> > subtract(double val, vector<vector<double> > vec) {
            vector<vector<double> > res(vec.size(), vector<double>(vec[0].size()));
            for(int i = 0; i < vec.size(); i ++) {
                for(int j = 0; j < vec[i].size(); j ++) {
                    res[i][j] = val - vec[i][j];
                }
            }
            return res;
        }
        vector<vector<double> > multiply(vector<vector<double> > v1, vector<vector<double> > v2) {
            vector<vector<double> > res(v1.size(), vector<double>(v1[0].size()));
            for(int i = 0; i < v1.size(); i ++) {
                for(int j = 0; j < v1[i].size(); j ++) {
                    res[i][j] = v1[i][j] * v2[i][j];
                }
            }
            return res;
        }
        vector<vector<double> > multiply(double val, vector<vector<double> > vec) {
            vector<vector<double> > res(vec.size(), vector<double>(vec[0].size()));
            for(int i = 0; i < vec.size(); i ++) {
                for(int j = 0; j < vec[i].size(); j ++) {
                    res[i][j] = val * vec[i][j];
                }
            }
            return res;
        }
        vector<vector<double> > add(vector<vector<double> > v1, vector<vector<double> > v2) {
            vector<vector<double> > res(v1.size(), vector<double>(v1[0].size()));
            for(int i = 0; i < v1.size(); i ++) {
                for(int j = 0; j < v1[i].size(); j ++) {
                    res[i][j] = v1[i][j] + v2[i][j];
                }
            }
            return res;
        }
        vector<vector<double> > transpose(vector<vector<double> > vec) {
            vector<vector<double> > res(vec[0].size(), vector<double>(vec.size()));
            for(int i = 0; i < vec.size(); i ++) {
                for(int j = 0; j < vec[i].size(); j ++) {
                    res[j][i] = vec[i][j];
                }
            }
            return res;
        }
};
class NeuralNetwork: public Numerical {
    private:
        int inputNodes, hiddenNodes, outputNodes;
        double learningRate;
        vector<vector<double> > wih, who;
    
    public:
        vector<vector<double> > activationFunction(vector<vector<double> > vec) {
            vector<vector<double> > res(vec.size(), vector<double>(vec[0].size()));
            for(int i = 0; i < vec.size(); i ++) {
                for(int j = 0; j < vec[i].size(); j ++) {
                    res[i][j] = 1.0 / (1 + exp(-vec[i][j]));
                }
            }
            return res;
        }
        void train(vector<vector<double> > inputs, vector<vector<double> > targets) {
            vector<vector<double> > hiddenInputs = dot(wih, inputs);
            vector<vector<double> > hiddenOutputs = activationFunction(hiddenInputs);
            vector<vector<double> > finalInputs = dot(who, hiddenOutputs);
            vector<vector<double> > finalOutputs = activationFunction(finalInputs);
            vector<vector<double> > outputsErrors = subtract(targets, finalOutputs);
            vector<vector<double> > hiddenErrors = dot(transpose(who), outputsErrors);
            who = add(who, multiply(learningRate, dot(multiply(outputsErrors, multiply(finalOutputs, subtract(1.0, finalOutputs))), transpose(hiddenOutputs))));
            wih = add(wih, multiply(learningRate, dot(multiply(hiddenErrors, multiply(hiddenOutputs, subtract(1.0, hiddenOutputs))), transpose(inputs))));
        }
        vector<vector<double> > query(vector<vector<double> > inputs) {
            vector<vector<double> > hiddenInputs = dot(wih, inputs);
            vector<vector<double> > hiddenOutputs = activationFunction(hiddenInputs);
            vector<vector<double> > finalInputs = dot(who, hiddenOutputs);
            vector<vector<double> > finalOutputs = activationFunction(finalInputs);

            return finalOutputs;
        }
        NeuralNetwork() {}
        NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, double lr) {
            inputNodes = inputnodes;
            hiddenNodes = hiddennodes;
            outputNodes = outputnodes;
            learningRate = lr;
            randomNormal(wih, hiddennodes, inputnodes, 1.0 / sqrt(hiddennodes));
            randomNormal(who, outputnodes, hiddennodes, 1.0 / sqrt(outputnodes));
        }
};
NeuralNetwork nn;

void training(string fileName) {
    vector<vector<double> > X(784, vector<double>(1));
    ifstream rf(fileName);
    string line;
    while(getline(rf, line)) {
        vector<vector<double> > y(10, vector<double>(1, 0.01));
        string spilt;
        istringstream readstr(line);
        for(int i = 0; i < 785; i ++) {
            getline(readstr, spilt, ',');
            int val = atoi(spilt.c_str());
            if(i == 0) y[val][0] = 0.99;
            else X[i - 1][0] = val * 1.0 / 255 * 0.99 + 0.01;
        }
        nn.train(X, y);
    }
}
void testing(string fileName) {
    vector<vector<double> > X(784, vector<double>(1));
    double score = 0;
    ifstream rf(fileName);
    string line;
    while(getline(rf, line)) {
        int corrertLabel, label;
        double x = 0;
        string spilt;
        istringstream readstr(line);
        for(int i = 0; i < 785; i ++) {
            getline(readstr, spilt, ','); //根据逗号分割字符串
            int val = atoi(spilt.c_str());
            if(i == 0) corrertLabel = val;
            else X[i - 1][0] = val * 1.0 / 255 * 0.99 + 0.01;
        }
        vector<vector<double> > res = nn.query(X);
        for(int i = 0; i < 10; i ++) {
            if(res[i][0] > x) {
                x = res[i][0];
                label = i;
            }
        }
        if(label == corrertLabel) score += 1;
    }
    //printf("%f\n", score / 10000);
    cout<<score / 10000<<"\n";
}

int main(){

    srand((int)time(0));
    
    int inputNodes = 784, hiddenNodes = 100, outputNodes = 10;
    double learningRate = 0.1;
    string trainDataFile = "data/mnist_train.csv";
    string testDataFile = "data/mnist_test.csv";

    nn = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate);
    
    training(trainDataFile);
    
    testing(testDataFile);
    
    return 0;
}