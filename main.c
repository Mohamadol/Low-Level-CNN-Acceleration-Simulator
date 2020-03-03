#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include"./layers/cnn.h"
#include"./layers/fcn.h"
#include "layers/pooling.h"
#include "./layers/relu.h"

#define sigmoid(x) 1/(1+exp(-x))

//Constants
const char main_image [14] = "./data/i4.txt";
const char main_weights[10][14] = { "./data/w1.txt", "./data/w2.txt", "./data/w3.txt" , "./data/w4.txt" , "./data/w5.txt" , "./data/w6.txt" , "./data/w7.txt" , "./data/w8.txt"  , "./data/w9.txt", "./data/b2.txt" };
const char labels[10][20] = { "airplane" , "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

//Function Declarations
double *read_values(char file_name[7], int size);



//Main
int main() {



	//*******Layer 1************
	int img_size[3] = {32, 32, 3};
	double * img = read_values(main_image, 32*32*3);
	int w1_size[4] = {3, 3, 3, 32};
	double * w1	 = read_values(main_weights[0], 3*3*3*32);

	cnn * conv1 = cnn_init(img_size, w1_size, img, w1, 1, 1);
	cnn_convolve(conv1);

	free(img);
	free(w1);
	//printf("first checkpoint\n");

	relu * relu1 = relu_init(conv1->output_w * conv1->output_h * conv1->output_c, conv1->output);
	relu_process(relu1);

	//*******EndOfLayer1********


	//*******Layer 2************
	int in2_size[3] = { conv1->output_w, conv1->output_h, conv1->output_c};
	double *in2 = relu1->output;
	int w2_size[4] = {3, 3, 32, 32};
	double * w2 = read_values(main_weights[1], 3 * 3 * 32 * 32);
	/*
	printf("%0.5f\n", *(w2 ));
	printf("%0.5f\n", *(w2 + 1)); 
	printf("%0.5f\n", *(w2 + 2));
	*/

	cnn * conv2 = cnn_init(in2_size, w2_size, in2, w2, 1, 1);
	cnn_convolve(conv2);

	free(in2);
	free(w2);

	//printf("second checkpoint\n");
	relu * relu2 = relu_init(conv2->output_w * conv2->output_h * conv2->output_c, conv2->output);
	relu_process(relu2);

	/*
	printf("Shape of output is %d   %d  %d", conv2->output_c, conv2->output_h, conv2->output_c);
	int tmp_i;
	int tmp_j;
	for (tmp_i = 0; tmp_i < conv2->output_h; tmp_i++) {
		printf("\n\n\n");
		for (tmp_j = 0; tmp_j < conv2->output_w; tmp_j++) {
			printf("%0.3f ", *(relu2->output + tmp_j + tmp_i * conv2->output_w));
		}
	}*/
	//*******EndOfLayer2********
	
	//*******PoolingLayer*******
	int pool1_size[3] = { conv2->output_w, conv2->output_h, conv2->output_c};
	pooling *pool1 = pooling_init(pool1_size, relu2->output);
	pooling_process(pool1);
	free(conv2->output);
	//printf("third checkpoint\n");
	/*
	printf("Shape of output is %d   %d  %d", pool1->output_w,pool1->output_h,pool1->output_c);
	int tmp_i;
	int tmp_j;
	for (tmp_i = 0; tmp_i < pool1->output_h; tmp_i++) {
		printf("\n\n\n");
		for (tmp_j = 0; tmp_j < pool1->output_w; tmp_j++) {
			printf("%0.3f ", *(pool1->output + tmp_j + tmp_i * pool1->output_w));
		}
	}
	*/
	//*******EndOfPooling*******

	//*******Layer 3************
	int in3_size[3] = { pool1->output_w, pool1->output_h, pool1->output_c };
	double *in3 = pool1->output;
	int w3_size[4] = { 3, 3, 32, 64};
	double * w3 = read_values(main_weights[2], 3 * 3 * 32 * 64);

	cnn * conv3 = cnn_init(in3_size, w3_size, in3, w3, 1, 1);
	cnn_convolve(conv3);
	free(in3);
	free(w3);
	//printf("fourth checkpoint\n");
	relu * relu3 = relu_init(conv3->output_w * conv3->output_h * conv3->output_c, conv3->output);
	relu_process(relu3);
	//*******EndOfLayer3********
	//*******Layer 4************
	int in4_size[3] = { conv3->output_w, conv3->output_h, conv3->output_c };
	double *in4 = relu3->output;
	int w4_size[4] = {3, 3, 64, 64};
	double * w4 = read_values(main_weights[3], 3 * 3 * 64 * 64);

	cnn * conv4 = cnn_init(in4_size, w4_size, in4, w4, 1, 1);
	cnn_convolve(conv4);
	free(in4);
	free(w4);
	//printf("fifth checkpoint\n");
	relu * relu4 = relu_init(conv4->output_w * conv4->output_h * conv4->output_c, conv4->output);
	relu_process(relu4);
	//*******EndOfLayer4********
	//*******PoolingLayer*******
	int pool2_size[3] = { conv4->output_w, conv4->output_h, conv4->output_c };
	pooling *pool2 = pooling_init(pool2_size, relu4->output);
	pooling_process(pool2);
	free(conv4->output);
	//printf("sixth checkpoint\n");

	//*******EndOfPooling*******

	//*******Layer 5************
	int in5_size[3] = { pool2->output_w, pool2->output_h, pool2->output_c };
	double *in5 = pool2->output;
	int w5_size[4] = { 3, 3, 64, 128};
	double * w5 = read_values(main_weights[4], 3 * 3 * 128 * 64);

	cnn * conv5 = cnn_init(in5_size, w5_size, in5, w5, 1, 1);
	cnn_convolve(conv5);
	free(in5);
	free(w5);
	//printf("seventh checkpoint\n");
	relu * relu5 = relu_init(conv5->output_w * conv5->output_h * conv5->output_c, conv5->output);
	relu_process(relu5);
	//*******EndOfLayer5********
	//*******Layer 6************
	int in6_size[3] = { conv5->output_w, conv5->output_h, conv5->output_c };
	double *in6 = relu5->output;
	int w6_size[4] = { 3, 3, 128, 128};
	double * w6 = read_values(main_weights[5], 3 * 3 * 128 * 128);
	//printf("%0.6f", *(w6 + 1));

	cnn * conv6 = cnn_init(in6_size, w6_size, in6, w6, 1, 1);
	cnn_convolve(conv6);
	free(in6);
	free(w6);
	//printf("eight checkpoint\n");
	relu * relu6 = relu_init(conv6->output_w * conv6->output_h * conv6->output_c, conv6->output);
	relu_process(relu6);
	//*******EndOfLayer6********
	//*******PoolingLayer*******
	int pool3_size[3] = { conv6->output_w, conv6->output_h, conv6->output_c };
	pooling *pool3 = pooling_init(pool3_size, relu6->output);
	pooling_process(pool3);
	free(conv6->output);
	//printf("ninth checkpoint\n");
	/*
	printf("size of intermediate activations are %d %d %d", pool3->output_w, pool3->output_h, pool3->output_c);
	int tmp_i;
	int tmp_j;
	for (tmp_i = 0; tmp_i < pool3->output_h; tmp_i++) {
		printf("\n\n\n");
		for (tmp_j = 0; tmp_j < pool3->output_w; tmp_j++) {
			printf("%0.3f ", *(pool3->output + tmp_j + tmp_i * pool3->output_w + 2 * pool3->output_w * pool3->output_h));
		}
	}
	*/
	//*******EndOfPooling*******
	





	//******FCNLayers************
	double *in7 = pool3->output;
	double * w7 = read_values(main_weights[6], 2048*128);
	double * b7 = read_values(main_weights[7], 128);
	int reorder_sizes[3] = { 4, 4, 128};
	fcn * fcn1 = fcn_init(2048, 128, in7, w7, b7, 1, reorder_sizes);
	fcn_process(fcn1);

	/*
	printf("begins\n");
	int fcn_i;
	for (fcn_i = 0; fcn_i < 128; fcn_i++) {
		printf("%0.5f\n", *(fcn1->output + fcn_i));
	}
	*/
	

	relu * relu7 = relu_init(128, fcn1->output);

	relu_process(relu7);
	//printf("tenth checkpoint\n");
	free(w7);
	free(b7);


	double *in8 = relu7->output;
	double * w8 = read_values(main_weights[8], 10 * 128);
	
	//printf("%0.5f\n", *(w8 + 9*128 + 127));
	//printf("%0.5f\n", *(w8 + 2 * 128));

	
	double * b8 = read_values(main_weights[9], 10);
	//printf("%0.5f\n", *(b8 + 2));
	//printf("%0.5f\n", *(b8 + 9));


	fcn * fcn2 = fcn_init(128, 10, in8, w8, b8, 0, NULL);
	fcn_process(fcn2);
	//printf("eleventh checkpoint\n");
	free(w8);
	free(b8);
	//*****************************

	//******Softmax Activation********
	int main_i = 0;
	int max = 0;
	double max_val = 0.0;
	double main_sum = 0.0;

	for (main_i = 0; main_i < fcn2->outputs_size; main_i++) {
		main_sum += exp(*(fcn2->output + main_i));
	}
	
	for (main_i = 0; main_i < fcn2->outputs_size; main_i++) {
		double main_tmp = exp(*(fcn2->output + main_i)) / main_sum;
		if ( main_tmp > max_val) {
			max_val = main_tmp;
			max = main_i;
		}
	}
	
	printf("Final Prediction is %s\n", labels[max]);
	


	system("pause");
	return 0;
}



//Functions
double * read_values(char file_name[7], int size) {
	
	int i = 0;
	double *buffer = (double *)malloc((size) * sizeof(double));

	FILE * file = fopen(file_name, "r");
	if (file == NULL) {
		printf("can't open %s", file_name);
		return 0;
	}

	while (fscanf(file, "%lf", buffer + i) == 1)
	{
		i++;
	}

	return buffer;
}

/*
void tests() {
	fcn_test();
	cnn_test();
	cnn_test_strided();
	pooling_test();
}
*/
