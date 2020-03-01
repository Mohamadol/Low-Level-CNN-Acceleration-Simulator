#ifndef FCN_H 
#define FCN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//Types Declaration
typedef struct {
	double *inputs;
	double *weights;
	double *bias;
	double *output;
	int input_size;
	int outputs_size;
}fcn;

//Function Declartion
fcn * fcn_init(int input_size, int output_size, double *input, double *weights, double *bias);
void fcn_process(fcn *layer);



#endif 
