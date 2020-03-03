#include "fcn.h"



fcn * fcn_init(int input_size, int output_size, double *input, double *weights, double *bias, char reorder, int * reorder_sizes) {

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

	layer->reorder = reorder;
	layer->reordr_sizes = reorder_sizes;

	return layer;
}




void reorder_input(fcn *layer) {
	int ch, w, h;
	int i = 0;

	int channels = *(layer->reordr_sizes + 2);
	int height = *(layer->reordr_sizes + 1);
	int width = *(layer->reordr_sizes);

	double * new_input = (double *)malloc(4 * 4 * 128 * sizeof(double));
	for (w = 0; w < width; w++) {
		for (h = 0; h < height; h++) {
			for (ch = 0; ch < channels; ch++) {
				*(new_input + i) = *(layer->inputs + height * width * ch + h * width + w);
				i++;
			}
		}
	}
	free(layer->inputs);
	layer->inputs = new_input;
}


void fcn_process(fcn *layer) {

	int in, out;
	double partial_sum = 0.0;
	double current_weight = 0.0;
	double current_input = 0.0;

	if (layer->reorder == 1) {
		reorder_input(layer);
	}


	for (out=0;out<layer->outputs_size;out++) {

		for (in=0;in<layer->input_size;in++) {
			current_weight = (*(layer->weights + out * layer->input_size + in));
			current_input = (*(layer->inputs + in));
			partial_sum += current_input * current_weight;
		}
		*(layer->output + out) = partial_sum + *(layer->bias + out);
		partial_sum = 0.0;
	}

	return;
}
