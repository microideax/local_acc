#ifndef ACC_CTRL_H
#define ACC_CTRL_H
#include <stdint.h>
//==============board level interface============//
#define DEVICE_H2C      "/dev/xdma0_h2c_1"
#define DEVICE_C2H      "/dev/xdma0_c2h_1"
#define DEVICE_CTRL		"/dev/xdma0_user"

#define MAP_SIZE (32*1024UL)
#define CHECK_TIME 1

//==============address==================//
#define PARA_MEM_CONV	0x00000000
#define PARA_MEM_POOL   0xC4000000
#define CTRL_REG 		0x0000




#define BIAS_MEM        0xC0000000
#define DATA_IN_MEM     0x01000000         
#define DATA_OUT_MEM    0x01400000
#define WEIGHT_MEM      0x01800000
#define POOLING_OUT_MEM 0x01C00000          


#define BIAS_OFFSET		0x00000000
#define DATA_IN_OFFSET  0x00040000
#define DATA_OUT_OFFSET 0x00050000
#define WEIGHT_OFFSET   0x00060000
#define POOLING_OUT_OFFSET  0x00070000

void write_para(uint32_t* para_list,int type);
void write_weight(short int weight[][32],int weight_length);
void write_bias(int* bias,int bias_length);
void write_data(short int feature[][32],int feature_length);
void start_process();
void read_data_conv(short int feature[][32],int feature_length);
void read_data_pooling(short int feature[][32],int feature_length);
void write_weight_bias(short int weight[][32],int weight_length,int *bias,int bias_length);
void quick_start_conv(uint32_t 	input_num,
					 uint32_t 	output_num,
					 uint32_t 	kernel_size,
					 uint32_t   feature_out_size,
					 uint32_t*  para_list,
					 short int  feature_out[][32],
					 int 	    feature_out_length
					 );
void quick_start_pooling(
					 uint32_t*  para_list,
					 short int  feature_out[][32],
				 	 int 	    feature_out_length
					);
#endif
