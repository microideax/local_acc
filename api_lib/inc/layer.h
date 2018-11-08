#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <iostream>
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
						 );
void pooling_layer_construct(uint32_t feature_in_size,
							uint32_t  input_num,
							uint32_t kernel_size,
							uint32_t feature_out_size,
							uint32_t stride,
							uint32_t padding,
							uint32_t act,
							short int  feature_out[][32],
						 	int        feature_out_length
							);
#endif