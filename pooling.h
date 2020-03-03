#ifndef POOLING_H 
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>



//Types Declaration
typedef struct {
	double	*inputs;
	int		input_w, input_h, input_c;
	double	*output;
	int		output_w, output_h, output_c;
}pooling;


//Function Declartion
pooling * pooling_init(int *input_size, double *input);

void pooling_process(pooling *layer);
double maximum(double a1, double a2, double a3, double a4);


#endif