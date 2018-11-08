#include "layer.h"
#include "acc_ctrl.h"
#include <iostream>
using namespace std;


void conv_layer_construct(uint32_t 	input_num,
						 uint32_t 	output_num,
						 uint32_t 	kernel_size,
						 uint32_t 	feature_in_size,
						 uint32_t 	feature_out_size,
						 uint32_t 	stride,
						 uint32_t 	padding,
						 uint32_t 	act,	
						 short int  feature_out[][32],
						 int        feature_out_length
						 )
{
	uint32_t para_list[16] = {0};
	para_list[0] = input_num;
	para_list[1] = kernel_size;
	para_list[2] = output_num;
	para_list[3] = feature_in_size;
	para_list[4] = feature_in_size;
	para_list[5] = feature_out_size;
	para_list[6] = feature_out_size;
	para_list[7] = stride;
	para_list[8] = padding;
	para_list[9] = act;
	para_list[10] = WEIGHT_OFFSET;
	para_list[11] = BIAS_OFFSET;
	para_list[12] = DATA_IN_OFFSET;
	para_list[13] = DATA_OUT_OFFSET;
	para_list[14] = 0;
	para_list[15] = 0;
	quick_start_conv(input_num,
				output_num,
			    kernel_size,
			    feature_out_size,
		        para_list,
				feature_out,
				feature_out_length
			   );
}

void pooling_layer_construct(uint32_t feature_in_size,
							uint32_t  input_num,
							uint32_t kernel_size,
							uint32_t feature_out_size,
							uint32_t stride,
							uint32_t padding,
							uint32_t act,
							short int  feature_out[][32],
						 	int        feature_out_length
							)
{		
	uint32_t para_list[16] = {0};
	para_list[0] = feature_in_size;
	para_list[1] = feature_in_size;
	para_list[2] = input_num;
	para_list[3] = kernel_size;
	para_list[4] = feature_out_size;
	para_list[5] = feature_out_size;
	para_list[6] = stride;
	para_list[7] = padding;
	para_list[8] = act;
	para_list[9] = DATA_OUT_OFFSET;
	para_list[10] = POOLING_OUT_OFFSET;
	para_list[11] = 0;
	para_list[12] = 0;
	para_list[13] = 0;
	para_list[14] = 0;
	para_list[15] = 1;

	quick_start_pooling(
						 para_list,
						 feature_out,
					 	 feature_out_length
						);
}


