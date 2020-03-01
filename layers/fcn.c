#include "fcn.h"



fcn * fcn_init(int input_size, int output_size, double *input, double *weights, double *bias) {

	fcn *layer;
	layer = (fcn *) malloc (sizeof(fcn));
	
	layer->input_size = input_size;
	layer->outputs_size = output_size;
	layer->inputs = input;
	layer->weights = weights;
	layer->bias = bias;
	layer->output = (double *)malloc(output_size * sizeof(double));
	//check for dynamic allocation errors 
	if (layer->output==NULL) {
		printf("couldn't allocate memory in FCN\n");
		return 0;
	}

	return layer;
}



void fcn_process(fcn *layer) {

	int in, out;
	double partial_sum = 0.0;

	for (out=0;out<layer->outputs_size;out++) {

		for (in=0;in<layer->input_size;in++) {

			 partial_sum += (*(layer->inputs + in)) * (*(layer->weights + (out*3) + in));

		}
		*(layer->output + out) = partial_sum + *(layer->bias + out);
		partial_sum = 0.0;
	}

	return;
}