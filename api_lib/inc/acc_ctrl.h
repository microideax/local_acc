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
#define PARA_MEM		0x00000000
#define BIAS_MEM		0x00001000
#define WEIGHT_MEM      0x10000000
#define CTRL_REG 		0x0000
#define DATA_IN_MEM     0x01000000
#define DATA_OUT_MEM    0x01800000

#define WEIGHT_OFFSET   0x00000000
#define BIAS_OFFSET		0x00000000
#define DATA_IN_OFFSET  0x00040000
#define DATA_OUT_OFFSET 0x00060000

void write_para(uint32_t* para_list);
void write_weight(short int weight[][32],int weight_length);
void write_data(short int feature[][32],int feature_length);
void start_process(uint32_t input_num,uint32_t output_num,uint32_t kernel_size,uint32_t feature_in_size);
void read_data(short int feature[][32],int feature_length);
void quick_start(uint32_t 	input_num,
				 uint32_t 	output_num,
				 uint32_t 	kernel_size,
				 uint32_t   feature_in_size,
				 uint32_t* para_list,
				 short int weight[][32],
				 int       weight_length,
				 short int feature_in[][32],
				 int       feature_in_length,
				 short int feature_out[][32],
				 int 	   feature_out_length
				 );
#endif
