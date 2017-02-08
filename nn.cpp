//============================================================================
// Name        : nn.cpp
// Author      : Hossein Mani
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <stdio.h>
//#include <array>
#include <vector>
#include <math.h>

using namespace std;

typedef vector<int> rowVector;
typedef vector<double> rowWeight;



int main() {
	ifstream _file("or.in"); // or, irisData, cubeData, xor2, majority

	if (!_file.is_open()) {
		cerr << "File can not be find or oppened";
		exit(1);
	}

	// Reading Data and put in a vector

	string line;
	vector<rowVector> from_text;
	vector<rowVector> x_mat;
	vector<rowVector> d_mat;
	vector<rowVector> x_mat_tr;
	vector<rowVector> t_mat;
	vector<rowWeight> w_mat;
	vector<rowVector> y_mat;

	vector<rowVector> test_mat;


	unsigned int num_trainData, num_testData, num_output, num_input;
	int elements;
	unsigned int ithCol;
	unsigned int ithRow = 0;
cout << "READ DATA FROM FILE "<<endl;
		while(_file.good())
		{
			while(getline(_file, line))
			{
				from_text.push_back(rowVector());
				istringstream streamL(line);
				ithCol = 0;
				while(streamL >> elements)
				{
					from_text.back().push_back(elements);
					ithCol++;
				}
				ithRow++;
			}
		}

cout << "EXTRACT AND STORE DATA "<<endl;

		num_input = (unsigned int) from_text[0][0];
		num_trainData = (unsigned int) from_text[1][0];
		num_output = (unsigned int)from_text[1][1]- num_input;
		num_testData = (unsigned int)from_text[num_trainData+2][0];

//		cout << "this function has " << num_input << ", inputs"<< endl;
//		cout << "this function has " << num_output << ", outputs"<< endl;
//		cout << "# of train Data sets are " << num_trainData <<  endl;
//		cout << "# of test Data sets are " << num_testData <<  endl;

//		Put Data and target in related matrices
// x_mat and t_mat
for (unsigned int i = 0; i < num_trainData; ++i) {
	x_mat.push_back(rowVector());
	t_mat.push_back(rowVector());
	for (unsigned int j = 0; j < num_input; ++j) {  // fill x
		int value = from_text[i+2][j];
		x_mat.back().push_back(value);
	}
	x_mat.back().push_back(-1); // append -1 to the end of x_mat


	for (unsigned int j = 0; j < num_output; ++j) { //fill t matrix
		int target = from_text[i+2][j+num_input];
		t_mat.back().push_back(target);
	}
}

for (unsigned int i = 0; i <= num_input; ++i) {
	w_mat.push_back(rowWeight());
	for (unsigned int j = 0; j < num_output; ++j) {
		double u = (double)rand() / RAND_MAX;
		w_mat.back().push_back((2.0*u-1)/10.0);
	}
}


//cout << "transpose x" << endl;
// Transpose x_mat;
for (unsigned int i = 0; i <= num_input ; ++i) {
	x_mat_tr.push_back(rowVector());
	for (unsigned int j = 0; j <num_trainData; ++j) {
		int x_ji = x_mat[j][i];
		x_mat_tr.back().push_back(x_ji);
	}
}

//cout << "x*w and apply transfer function" << endl;
//y = x*w and apply transfer function
	for (unsigned int i = 0; i < num_trainData; ++i) {
		y_mat.push_back(rowVector());
		for (unsigned int j = 0; j <num_output; ++j) {
			double sum = 0;
			for (unsigned int k = 0; k <= num_input; ++k) {
				int x_ik = x_mat[i][k];
				double w_kj = w_mat[k][j];
				sum += w_kj* (double) x_ik;
			}
			int y_ij = (sum >	0.0) ? 1 : 0;
			y_mat.back().push_back(y_ij);
		}
	}


//	cout << "d = xtr*(t-y)" << endl;
	//d = x_mat_tr*(t-y)
	for (unsigned int i = 0; i <= num_input; ++i) {
		d_mat.push_back(rowVector());
		for (unsigned int j = 0; j < num_output; ++j) {
			int sum = 0;
			for (unsigned int k = 0; k < num_trainData; ++k) {
				int x_tr_ik = x_mat_tr[i][k];
				int ty_kj = t_mat[k][j]-y_mat[k][j];
				sum += ty_kj*x_tr_ik;
			}
			d_mat.back().push_back(sum);
		}
	}


	//update weights
	for (unsigned int i = 0; i <= num_input; ++i) {
		for (unsigned int j = 0; j < num_output; ++j) {
			w_mat[i][j] += 0.1*(double)d_mat[i][j];
		}
	}

cout << "BEGIN TRAINING NN " << endl;
int error_check = 0;
while((error_check < 100)) {//Do the loop
		//y = x*w and apply transfer function
		for (unsigned int i = 0; i < num_trainData; ++i) {
			for (unsigned int j = 0; j <num_output; ++j) {
				double sum = 0;
				for (unsigned int k = 0; k <= num_input; ++k) {
					int x_ik = x_mat[i][k];
					double w_kj = w_mat[k][j];
					sum += w_kj* (double) x_ik;
				}
				int y_ij = (sum >	0.0) ? 1 : 0;
				y_mat[i][j] = y_ij;
			}
		}


		//d = x_mat_tr*(t-y)
		for (unsigned int i = 0; i <= num_input; ++i) {
			for (unsigned int j = 0; j < num_output; ++j) {
				int sum = 0;
				for (unsigned int k = 0; k < num_trainData; ++k) {
					int x_tr_ik = x_mat_tr[i][k];
					int ty_kj = t_mat[k][j]-y_mat[k][j];
					sum += ty_kj*x_tr_ik;
				}
				d_mat[i][j] = (sum);
			}
		}

		//update weights
		for (unsigned int i = 0; i <= num_input; ++i) {
			for (unsigned int j = 0; j < num_output; ++j) {
				w_mat[i][j] += 0.1*(double)d_mat[i][j];
			}
		}
		error_check++;
};



// TEST DATA
cout << "BEGIN TESTING" << endl;

for (unsigned int i = 0; i < num_testData; ++i) {
	test_mat.push_back(rowVector());
	for (unsigned int j = 0; j < num_input; ++j) {  // fill x
		int value = from_text[i+3+num_trainData][j];
		test_mat.back().push_back(value);
	}
	test_mat.back().push_back(-1); // append -1 to the end of x_mat

	for (unsigned int j = 0; j <= num_input; ++j) {  // fill x
		}

}


//cout << "test_mat*w and apply transfer function" << endl;
//y = test_mat*w and apply transfer function
for (unsigned int i = 0; i < num_testData; ++i) {
	y_mat.push_back(rowVector());
	for (unsigned int j = 0; j <num_output; ++j) {
		double sum = 0;
		for (unsigned int k = 0; k <= num_input; ++k) {
			int x_ik = test_mat[i][k];
			double w_kj = w_mat[k][j];
			sum += w_kj * (double) x_ik;
		}
		int y_ij = (sum >0.0) ? 1 : 0;
		y_mat.back().push_back(y_ij);
	}
}



// DISPLAY RESULTS:
for (unsigned int i = 0; i < num_testData; ++i) {

	for (unsigned int j = 0; j <num_input; ++j) {
		cout << test_mat[i][j] << " ";
	}

	cout << "| ";
	for (unsigned int j = 0; j <num_output; ++j) {
		cout << y_mat[i][j] << " ";
	}
	cout << endl;
}

	return 0;
}























//	ofstream _ofile;
//	_ofile.open ("test.txt", ios_base::app);
//	if (_file.is_open()) {
//		_ofile << "It should be output" << endl;
//		_ofile.close();
//	}


//void CreatMat(vector<rowVector> &x_mat, unsigned int num_traintData, unsigned int num_inputs, unsigned int index_row, unsigned int index_col, vector<rowVector> &from_text ){
//for (unsigned int i = 0; i < num_traintData; ++i) {
//	x_mat.push_back(rowVector());
//	for (unsigned int j = 0; j < num_inputs; ++j) {
//		int value = from_text[i][j];
//		x_mat.back().push_back(value);
//	}
//}
//}



