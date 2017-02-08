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

void built_0_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn);

void built_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn, unsigned int rindx, unsigned int cindx, vector<rowVector>& i_mat, unsigned int append_bias);
//void add_bias(vector<rowVector>& o_mat, unsigned int rn);
void built_w_mat(vector<rowWeight>& o_mat, unsigned int rn, unsigned int cn);
void transpose_mat(vector<rowVector>& x_mat, unsigned int rn, unsigned int cn, vector<rowVector>& x_t_mat);
void mult_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowWeight>& b_mat, unsigned int rn_b, unsigned int cn_b,vector<rowVector>& c_mat);
void mult_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowVector>& b_mat, unsigned int rn_b, unsigned int cn_b,vector<rowVector>& c_mat);
void add_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowVector>& b_mat, unsigned int rn_b, unsigned int cn_b,vector<rowVector>& c_mat);
void neg_mat(vector<rowVector>& x_mat, unsigned int rn, unsigned int cn, vector<rowVector>& neg_x_mat);

void print_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn);



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

// x_mat and t_mat
		unsigned int append_bias = 1;
		built_mat(x_mat, num_trainData, num_input, 2, 0, from_text, append_bias);

		append_bias = 0;
		built_mat(t_mat, num_trainData, num_output, 2, num_input, from_text, append_bias);

		built_w_mat(w_mat, num_input+1, num_output); // Do not forget to add extra row for bias

		built_0_mat(y_mat, num_trainData, num_output);
		mult_mat(x_mat, num_trainData, num_input+1, w_mat, num_input+1, num_output, y_mat);

		vector<rowVector> neg_y_mat;
		built_0_mat(neg_y_mat, num_trainData, num_output);
		neg_mat(y_mat, num_trainData, num_output, neg_y_mat);

		vector<rowVector> temp_error_mat;
		built_0_mat(temp_error_mat, num_trainData, num_output);
		add_mat(t_mat, num_trainData, num_output, neg_y_mat, num_trainData, num_output, temp_error_mat);

		built_0_mat(x_mat_tr, num_input+1, num_trainData);
		transpose_mat(x_mat, num_trainData, num_input+1, x_mat_tr);



		built_0_mat(d_mat, num_input+1, num_output);
		mult_mat(x_mat_tr, num_input+1, num_trainData, temp_error_mat, num_trainData, num_output, d_mat);


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
	mult_mat(x_mat, num_trainData, num_input+1, w_mat, num_input+1, num_output, y_mat);

	neg_mat(y_mat, num_trainData, num_output, neg_y_mat);

	add_mat(t_mat, num_trainData, num_output, neg_y_mat, num_trainData, num_output, temp_error_mat);

	mult_mat(x_mat_tr, num_input+1, num_trainData, temp_error_mat, num_trainData, num_output, d_mat);

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

append_bias = 1;
built_mat(test_mat, num_testData, num_input, 3+num_trainData, 0, from_text, append_bias);

mult_mat(test_mat, num_testData, num_input+1, w_mat, num_input+1, num_output, y_mat);




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


/*MY FUNCTIONS BODY*/

void built_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn, unsigned int rindx, unsigned int cindx, vector<rowVector>& i_mat, unsigned int append_bias){
	for (unsigned int i = 0; i < rn; ++i) {
		o_mat.push_back(rowVector());
		for (unsigned int j = 0; j < cn; ++j) {  // fill x
			int value = i_mat[i+rindx][j+cindx];
			o_mat.back().push_back(value);
		}
		if (append_bias) {
			o_mat.back().push_back(-1);
		}
	}

};


void add_bias(vector<rowVector>& o_mat, unsigned int rn){
	for (unsigned int i = 0; i < rn; ++i) {
		o_mat.push_back(rowVector());
		o_mat.back().push_back(-1); // append -1 to the end of x_mat
	}
};



void built_w_mat(vector<rowWeight>& o_mat, unsigned int rn, unsigned int cn){
	for (unsigned int i = 0; i < rn; ++i) { // Considers bias automatically
		o_mat.push_back(rowWeight());
		for (unsigned int j = 0; j < cn; ++j) {
			double u = (double)rand() / RAND_MAX;
			o_mat.back().push_back((2.0*u-1)/10.0);
		}
	}
};


void print_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn){
	for (unsigned int i = 0; i < rn; ++i) {

		for (unsigned int j = 0; j <cn; ++j) {
			cout << o_mat[i][j] << " ";
		}
		cout << endl;
	}
};


void transpose_mat(vector<rowVector>& x_mat, unsigned int rn, unsigned int cn, vector<rowVector>& x_t_mat){
	for (unsigned int i = 0; i < cn ; ++i) {
		for (unsigned int j = 0; j <rn; ++j) {
			int x_ji = x_mat[j][i];
			x_t_mat[i][j] = x_ji;
		}
	}
};

void mult_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowWeight>& b_mat, unsigned int rn_b, unsigned int cn_b, vector<rowVector>& c_mat){

	for (unsigned int i = 0; i < rn_a; ++i) {
			for (unsigned int j = 0; j <cn_b; ++j) {
				double sum = 0;
				for (unsigned int k = 0; k <cn_a; ++k) {
					int a_ik = a_mat[i][k];
					double b_kj = b_mat[k][j];
					sum += b_kj* (double) a_ik;
				}
				int y_ij = (sum >	0.0) ? 1 : 0;
				c_mat[i][j] = y_ij;
			}
		}

};

void mult_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowVector>& b_mat, unsigned int rn_b, unsigned int cn_b, vector<rowVector>& c_mat){

	for (unsigned int i = 0; i < rn_a; ++i) {
			for (unsigned int j = 0; j <cn_b; ++j) {
				double sum = 0;
				for (unsigned int k = 0; k <cn_a; ++k) {
					int a_ik = a_mat[i][k];
					double b_kj = b_mat[k][j];
					sum += b_kj* a_ik;
				}
				c_mat[i][j] = sum;
			}
		}

};

void neg_mat(vector<rowVector>& x_mat, unsigned int rn, unsigned int cn, vector<rowVector>& neg_x_mat){
	for (unsigned int i = 0; i < rn; ++i) {
		for (unsigned int j = 0; j <cn; ++j) {
			int val = -1*x_mat[i][j];
			neg_x_mat[i][j] = val;
		}
	}
};


void add_mat(vector<rowVector>& a_mat, unsigned int rn_a, unsigned int cn_a, vector<rowVector>& b_mat, unsigned int rn_b, unsigned int cn_b,vector<rowVector>& c_mat){
	for (unsigned int i = 0; i < rn_a; ++i) {
		for (unsigned int j = 0; j <cn_a; ++j) {
			int val = a_mat[i][j]+b_mat[i][j];
			c_mat[i][j] = val;;
		}
	}
};

void built_0_mat(vector<rowVector>& o_mat, unsigned int rn, unsigned int cn){
	for (unsigned int i = 0; i < rn; ++i) { // Considers bias automatically
		o_mat.push_back(rowVector());
		for (unsigned int j = 0; j < cn; ++j) {
			o_mat.back().push_back(0);
		}
	}
};


//	ofstream _ofile;
//	_ofile.open ("test.txt", ios_base::app);
//	if (_file.is_open()) {
//		_ofile << "It should be output" << endl;
//		_ofile.close();
//	}





