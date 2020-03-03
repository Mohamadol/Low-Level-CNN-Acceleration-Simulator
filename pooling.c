#include "pooling.h"


pooling * pooling_init(int *input_size, double *input) {

	pooling *layer = (pooling *)malloc(sizeof(pooling));
	if (layer == NULL) {
		printf("Problem allocating dynamic memory in pooling\n");
		return 0;
	}

	layer->inputs = input;
	layer->input_c = *(input_size + 2);	//channels
	layer->input_h = *(input_size + 1);	//Height
	layer->input_w = *(input_size);		//Width

	//if (layer->input_h % 2 != 0)
		//pad_input();

	layer->output_c = *(input_size + 2);
	layer->output_h = (*(input_size + 1))/2;
	layer->output_w = (*(input_size))/2;
	layer->output = (double *)malloc((layer->output_w * layer->output_h * layer->output_c) * sizeof(double));
	if (layer->output == NULL) {
		printf("couldn't allocate memory in CNN\n");
		return 0;
	}

}



void pooling_process(pooling *layer) {

	int channel, height, width, output_index;
	output_index = 0;
	double in1, in2, in3, in4;

	for (channel = 0; channel < layer->input_c; channel++) {
		for (height = 0; height < layer->input_h; height+=2) {
			for (width = 0; width < layer->input_w; width+=2) {
				in1 = *(layer->inputs + channel * layer->input_h * layer->input_w + height * layer->input_w + width);
				in2 = *(layer->inputs + channel * layer->input_h * layer->input_w + height * layer->input_w + width+1);
				in3 = *(layer->inputs + channel * layer->input_h * layer->input_w + (height+1) * layer->input_w + width);
				in4 = *(layer->inputs + channel * layer->input_h * layer->input_w + (height+1) * layer->input_w + width + 1);
				*(layer->output + output_index) = maximum(in1, in2, in3, in4);
				output_index++;
				//printf("args are %0.3f   %0.3f   %0.3f   %03.f   max is = %0.3f", in1, in2, in3, in4, *(layer->output + output_index - 1));
			}
		}
	}

}

double maximum(double a1, double a2, double a3, double a4) {
	double result;

	result = (a1 < a2) ? a2 : a1;
	result = (result < a3) ? a3 : result;
	result = (result < a4) ? a4 : result;

	return result;
}