#ifndef CNN_H 
#define CNN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Types Declaration
typedef struct {
	double	*inputs;
	int		input_w, input_h, input_c;
	double	*kernel;
	int		kernel_w, kernel_h, kernel_ci, kernel_co;
	double	*output;
	int		output_w, output_h, output_c;
	int stride;
	int padding;
}cnn;

//Function Declartion
cnn * cnn_init(int *input_size, int *kernel_size, double *input, double *kernel, int stride, int padding);
void cnn_convolve(cnn *layer);
void cnn_pad(cnn * layer);

#endif