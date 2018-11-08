#include <caffe/caffe.hpp>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>


#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "layer_factory_impl.h"
#include "../../api_lib/inc/layer.h"
#include "../../api_lib/inc/acc_ctrl.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>



#define CNN_OPEN_BINARY(filename) open(filename, O_RDONLY)
#define CNN_OPEN_TXT(filename) open(filename, O_RDONLY)

using namespace caffe;
using namespace std;
using namespace cv;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
using boost::shared_ptr;

void create_net_from_caffe_prototxt(const std::string& caffeprototxt,int input_param[]);
void reload_weight_from_caffe_protobinary(const std::string& caffebinary,int input_param[]);
void software_validate_conv(int num_input,int num_output,int kernel_size,int stride,int feature_in_size, 
                       short int feature_in[][32],float feature_out[][96],short int weight[][32],int* bias);
void software_validate_pooling(short int feature_in[][32],short int feature_out[][32]);
const int my_num_input = 3;
const int my_num_ouput = 96;
const int kernel_size = 11;

short int weight[my_num_ouput * kernel_size * kernel_size][32] = {0};
int       bias[my_num_ouput] = {0};
short int feature_in[227*227][32] = {0};
short int test_temp[227*227][32] = {0};
short int feature_out[my_num_ouput / 3 * 55*55][32]={0};
short int pooling_out[3*27*27][32] = {0};
float feature_out1[55*55][96]={0};
short int pooling_out1[3*27*27][32] = {0};



int main(int argn, char* argv[])
{
    string caffemodel = argv[1];
    string prototxt   = argv[2];
    
    cout << "Read prototxt file from: " << prototxt << endl;
    cout << "Read caffemodel file from: " << caffemodel << endl;
    Net<float> net(prototxt,TEST);
    net.CopyTrainedLayersFrom(caffemodel); 

    vector<boost::shared_ptr<Blob<float>>> params=net.params(); 

    for(int i=0;i<params.size();++i) 
        cout<<i<<":"<<params[i]->shape_string()<<endl; 




    cout << "width:" << params[2]->shape(2)<<endl;
    cout << "height" << params[2]->shape(3)<<endl;
    cout << "num"    << params[2]->shape(0)<<endl;
    cout << "??:"    << params[2]->shape(1)<<endl; 


/*    int input_param[] = {0, 0};
    create_net_from_caffe_prototxt(argv[2], input_param);
    reload_weight_from_caffe_protobinary(argv[1], input_param);*/

    // Mat frame = imread("lena.png"); 
    // Mat smallImage;
    // Mat splitImage[3];
    // Mat outmergeImage;
    // Mat outputFeature[96];
    // int outmergeImageRows = 55 * 6 + 5 * 5;
    // int outmergeImageCols = 55 * 16 + 5 * 15;
    
    // write_weight_bias(weight,kernel_size*kernel_size*64*my_num_ouput,bias,my_num_ouput*4);

    // resize(frame,smallImage,Size(227,227));
    // split(smallImage,splitImage);
    // // imshow("test",frame);
    // imshow("smallImage",smallImage);

    // for(int i = 0 ; i < 96; i++)
    //     outputFeature[i].create(Size(55,55),splitImage[0].type());
    // outmergeImage.create(outmergeImageRows,outmergeImageCols,outputFeature[0].type());

    // for(int i = 0 ; i < 227*227; i++)
    //     for(int j = 0 ; j < my_num_input;j++)
    //         feature_in[i][j]=(short int)((splitImage[j].at<uchar>(i/227,i%227)) << 6);



    // write_data(feature_in,227*227*64);

    // conv_layer_construct(my_num_input,                      //input_num
    //                     my_num_ouput,                       //output_num
    //                     kernel_size,                        //kernel_size
    //                     227,                                //feature_in_size
    //                     55,                                 //feature_out_size
    //                     4,                                  //stride
    //                     0,                                  //padding
    //                     1,                                  //act
    //                     feature_out,                        //feature_out
    //                     55*55*64*(my_num_ouput/32)          //feature_out_length
    //                     );
    // software_validate_conv(my_num_input,my_num_ouput,kernel_size,4,227,feature_in,feature_out1,weight,bias);
    // for(int k = 0 ; k < my_num_ouput; k++)
    // {
    //     cout << "*******output" << k << "                                " <<  "*******software output" <<endl;
    //     for(int i = 50 ; i < 55; i++)
    //     {
    //         for(int j = 50; j < 55; j++)
    //             cout <<setw(8)<<(feature_out[k/32 * 55 * 55 + i * 55 + j][k%32]/64.0) <<" ";
    //         cout << "  |  ";
    //         for(int j = 50; j < 55; j++)  
    //             cout <<setw(8)<<(feature_out1[i*55+j][k]) << " ";

    //         cout << endl;
    //     }
    // }





    // pooling_layer_construct(55,                //feature_in_size
    //                         96,                //input_num
    //                         3,                 //kernel_size
    //                         27,                //feature_out_size
    //                         2,                 //stride
    //                         0,                 //padding
    //                         1,                 //act
    //                         pooling_out,       //feature_out
    //                         27*27*3*64         //feature_out_length
    //                         );

    // software_validate_pooling(feature_out,pooling_out1);


    // for(int i = 0 ; i < 10; i++)
    // {
    //     for(int j = 0 ; j < 10; j++)
    //         cout << setw(6) << (pooling_out[i*27+j][0] /64.0 )<<" ";
    //     cout << endl;
    // }
    // cout << "=========="<< endl;


    // for(int i = 0 ; i < 10; i++)
    // {
    //     for(int j = 0 ; j < 10; j++)
    //         cout << setw(6) << (pooling_out1[i*27+j][0]/64.0)<<" ";
    //     cout << endl;
    // }



    // int sum = 0;
    // cout << "$$$$$$$$$$$$$$$$"<< endl;
    // for(int i = 0 ; i < 27; i++)
    // {
    //     for(int j = 0 ; j < 27; j++)
    //         if(pooling_out[i*27+j][0]!=pooling_out1[i*27+j][0])
    //         {
    //             cout << "i = " << i << "     j = " << j << endl;
    //             sum++;
    //         }
    //  }




    waitKey(0);


   /* if(capture.isOpened())
    {
        cout << "success"<< endl;
        capture >> frame;
        resize(frame,smallImage,Size(227,227));
        imshow("test",smallImage);
        split(smallImage,splitImage);
        for(int i = 0 ; i < 96; i++)
            outputFeature[i].create(Size(55,55),splitImage[0].type());
        outmergeImage.create(outmergeImageRows,outmergeImageCols,outputFeature[0].type());
    }


    while(capture.isOpened())
    {
        capture >> frame;
        resize(frame,smallImage,Size(227,227));
        imshow("test",smallImage);
        split(smallImage,splitImage);
        for(int i = 0 ; i < 227*227; i++)
            for(int j = 0 ; j < my_num_input;j++)
                feature_in[i][j]=(short int)((splitImage[j].at<uchar>(i/227,i%227)) << 6);

        write_data(feature_in,227*227*64);

        conv_layer_construct(my_num_input,                      //input_num
                            my_num_ouput,                       //output_num
                            kernel_size,                        //kernel_size
                            227,                                //feature_in_size
                            55,                                 //feature_out_size
                            4,                                  //stride
                            0,                                  //padding
                            1,                                  //act
                            feature_out,                        //feature_out
                            55*55*64*(my_num_ouput/32)          //feature_out_length
                            );


        // for(int i = 0 ; i < 10; i++)
        // {
        //     for(int j = 0 ; j < 10; j++)
        //         cout << setw(6)<< (feature_out[i*55+j][0] >> 6)<< " ";
        //     cout << endl;
        // }
       

        // waitKey(0);
        // while(1);
        // cout << "TEST!"<< endl;

        // software_validate_pooling(feature_out,pooling_out1);

        // for(int i = 0 ; i < 10; i++)
        // {
        //     for(int j = 0 ; j < 10; j++)
        //         cout << setw(6) << pooling_out1[i*27+j][0]<<" ";
        //     cout << endl;
        // }



        // cout << "hardware pooling out:"<<endl;

        // // write_data(feature_out,55*55*64*3);

        // pooling_layer_construct(55,
        //                         96,
        //                         3,
        //                         27,
        //                         2,
        //                         0,
        //                         0,
        //                         pooling_out,
        //                         27*27*64*3
        //                     );



        // for(int i = 0 ; i < 10; i++)
        // {
        //     for(int j = 0 ; j < 10; j++)
        //         cout << setw(6) << pooling_out[i*27+j][0]<<" ";
        //     cout << endl;
        // }

        // int sum = 0;
        // cout << "$$$$$$$$$$$$$$$$"<< endl;
        // for(int i = 0 ; i < 27; i++)
        // {
        //     for(int j = 0 ; j < 27; j++)
        //         if(pooling_out[i*27+j][0]!=pooling_out1[i*27+j][0])
        //         {
        //             cout << "i = " << i << "     j = " << j << endl;
        //             sum++;
        //         }

           
        // }

        // cout << sum<<endl;
        // while(1);

        // for(int i = 0 ; i < 96; i++)
        //     for(int j = 0 ; j < 55 * 55; j++)
        //         outputFeature[i].at<uchar>(j/55,j%55) = (feature_out[j][i]/64.0) <= 255 ? (feature_out[j][i]/64.0) : 255;  

        // for(int i = 0 ; i < 96 ; i++)
        //     outputFeature[i].copyTo(outmergeImage(Rect((i%16)*(55+5),(i/16)*(55+5),55,55)));  
       
        // imshow("outputFeature",outmergeImage);






        software_validate(my_num_input,my_num_ouput,kernel_size,4,227,feature_in,feature_out1,weight,bias);
        
        cout << "================feature0================"<<endl;
        for(int i = 50; i < 55; i++)
        {
            for(int j = 50; j < 55; j++)
                cout << setw(8)<<int(outputFeature[0].at<uchar>(i,j))<< " ";
            cout << endl;   
        }

        cout << "================feature1================"<<endl;
        for(int i = 50; i < 55; i++)
        {
            for(int j = 50; j < 55; j++)
                cout << setw(8)<<int(outputFeature[1].at<uchar>(i,j))<< " ";
            cout << endl;   
        }
        cout << "================feature2================"<<endl;
        for(int i = 50; i < 55; i++)
        {
            for(int j = 50; j < 55; j++)
                cout << setw(8)<<int(outputFeature[2].at<uchar>(i,j))<< " ";
            cout << endl;   
        }
        for(int k = 0 ; k < my_num_ouput; k++)
        {
            cout << "*******output" << k << "                                " <<  "*******software output" <<endl;
            for(int i = 50 ; i < 55; i++)
            {
                for(int j = 50; j < 55; j++)
                    cout <<setw(8)<<(feature_out[k/32 * 55 * 55 + i * 55 + j][k%32]/64.0) <<" ";
                cout << "  |  ";
                for(int j = 50; j < 55; j++)  
                    cout <<setw(8)<<(feature_out1[i*55+j][k]) << " ";

                cout << endl;
            }
        }
        while(1);




        char key = static_cast<char>(cvWaitKey(1));
            if(key == 27)
                break;
    }*/
    
    return 0;
}


void software_validate_conv(int num_input,int num_output,int kernel_size,int stride,int feature_in_size, 
                       short int feature_in[][32],float feature_out[][96],short int weight[][32],int* bias)
{
    int i,j,k,x,y,z,h;
    double temp;
    j = 0 ; k = 0;
    for(h = 0 ; h < num_output; h++)
    {
        z = 0;
        for(j = 0 ; j < feature_in_size - kernel_size + 1; j+=stride)
            for(k = 0 ; k < feature_in_size- kernel_size + 1; k+=stride)
            {
                temp = 0;
                for(i = 0 ; i < num_input; i++)
                    for(x = 0; x < kernel_size; x++)
                        for(y = 0 ; y < kernel_size; y++)
                        {
                            temp += (feature_in[(j * feature_in_size + k) +y+x*feature_in_size][i]/64.0) * (weight[h * kernel_size * kernel_size + x * kernel_size + y][i]/64.0);
                           /* cout <<"    x = " <<setw(2)<< x 
                                 <<"    y = " <<setw(2)<< y
                                 <<"    feature = " << setw(10)<<feature_in[ y+x*feature_in_size][0]/64 
                                 <<"    weight = " << setw(10)<<weight[x * kernel_size + y][0]/64.0
                                 <<"    temp0 = "  << setw(10)<<(feature_in[ y+x*feature_in_size][0]/64.0) * (weight[x * kernel_size + y][0]/64.0) 
                                 <<"    temp = "<<setw(8)<<temp<<endl; */
                           
                        }
                feature_out[z++][h] = ((temp + bias[h]/64.0)>0) ? (temp + bias[h]/64.0) : 0;
            }
    }
}


void software_validate_pooling(short int feature_in[][32],short int feature_out[][32])
{

    int i,j,k,x,y,z;
    int temp_array[9] = {0};
    int temp_max = 0;
    for(i = 0 ; i < 96; i++)
    {
        for(j = 0 ; j < 53; j+=2)
        {
            for(k = 0 ; k < 53; k+=2)
            {
                for(x = 0 ; x < 3; x++)
                    for(y = 0 ; y < 3; y++)
                        temp_array[x*3+y] = feature_in[i/32*55*55 + j*55 + k + (x*55+y)][i%32];

                    
                temp_max = temp_array[0];

                for(z = 1; z < 9; z++)
                    if(temp_array[z] > temp_max)
                        temp_max = temp_array[z];
                
                feature_out[i/32*27*27 + (j/2) * 27 + k/2][i%32] = temp_max;
               // cout << temp_max << endl;
               // cout << "++"<< temp_max<< endl;
                // for(z = 0 ; z < 9; z++)
                // {
                //    // temp_array[0] = 2;  
                //     cout << temp_array[z] <<endl;//temp_array[0]"" << " ";
                // }
                
            }
        }
    }
}







void load(const LayerParameter& src,int num_input,int num_output,int kernel_size) 
{
    int src_idx = 0;
    cout<<"weights: "<<endl;
    //load weight
    for (int o = 0; o < 100; o++) 
    {
        for (int i = 0; i < num_input; i++) 
        {
            for (int x = 0; x < kernel_size; x++) 
            {
                for(int y = 0 ; y < kernel_size; y++)
                   // weight[o * kernel_size * kernel_size +  x * kernel_size + y][i] = (short int)(src.blobs(0).data(src_idx++) * 64 );
                    cout<<setw(15)<<src.blobs(0).data(src_idx++)<<" ";
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
/*    for(int o = 0 ; o < 32; o++)
    {
        for (int i = 0; i < num_input; i++) 
        {
            for (int x = 0; x < kernel_size; x++) 
            {
                for(int y = 0 ; y < kernel_size; y++)
                    cout << setw(15)<<weight[o * kernel_size * kernel_size +  x * kernel_size + y][i]<<" ";//= (short int)(src.blobs(0).data(src_idx++) * 64);
                cout << endl;
            }
            cout << endl;
        }
    }*/
   //    cout<<setw(15)<<src.blobs(0).count()<<endl;
    cout<<""<<endl;

    if(src.convolution_param().bias_term()==false)
    {

    }else
    {
        cout<<"bias: "<<endl;
        //load bias
        for (int o = 0; o < num_output; o++) 
        {
            
            bias[o] = src.blobs(1).data(o) * 64;
            //cout<<bias[o]<<" ";
        }
        cout<<""<<endl;
    }
}

void read_proto_from_text(const std::string& prototxt,
                          google::protobuf::Message *message) 
{
    int fd = CNN_OPEN_TXT(prototxt.c_str());
    if (fd == -1) 
        cout<<"file not found: "<<prototxt<<endl;

    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);

    if (!google::protobuf::TextFormat::Parse(&input, message)) 
        cout<<"failed to parse"<<endl;
}

void create_net_from_caffe_net(const caffe::NetParameter& layer,int input_param[])
{
    cout << "create net from caffe net" << endl;
    caffe_layer_vector src_net(layer);
    if (layer.input_shape_size() > 0) 
    {
        input_param[0]  = static_cast<int>(layer.input_shape(0).dim(1));
        cout << "input_param[0]: " << input_param[0] << endl;
        input_param[1] = static_cast<int>(layer.input_shape(0).dim(2));
        cout << "input_param[1]: " << input_param[1] << endl;
    }
    else if (layer.layer(0).has_input_param()) 
    {
        cout << "input has layer param " << endl;
        input_param[0] = static_cast<int>(layer.layer(0).input_param().shape(0).dim(1));
        input_param[1] = static_cast<int>(layer.layer(0).input_param().shape(0).dim(2));
        cout << "input_param[0]: " << input_param[0] << endl;
        cout << "input_param[1]: " << input_param[1] << endl;
    }
}

void create_net_from_caffe_prototxt(const std::string& caffeprototxt,int input_param[])
{
    caffe::NetParameter np;
    read_proto_from_text(caffeprototxt, &np);
    cout <<"net_name from prototxt: "<<np.name()<<endl;
    create_net_from_caffe_net(np,input_param);
}

void read_proto_from_binary(const std::string& protobinary,
                            google::protobuf::Message *message) 
{
    int fd = CNN_OPEN_BINARY(protobinary.c_str());
    google::protobuf::io::FileInputStream rawstr(fd);
    google::protobuf::io::CodedInputStream codedstr(&rawstr);

    rawstr.SetCloseOnDelete(true);
    codedstr.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::max() / 2);

    if (!message->ParseFromCodedStream(&codedstr)) 
    {
        cout<<"failed to parse"<<endl;
    }
}

void reload_weight_from_caffe_net(const caffe::NetParameter& layer,int input_param[])
{
    caffe_layer_vector src_net(layer);
    int num_layers = src_net.size();
    cout << "src_net size: " << num_layers << endl;

    int num_input=input_param[0];
    int input_size=input_param[1];
    int num_output=0;
    for (int i = 0; i < src_net.size(); i++) 
    {
        int pad=0;
        int kernel_size=0;
        int stride=1;

        if(src_net[i].type()=="Convolution"||src_net[i].type()=="ConvolutionRistretto")
        {//get conv_layers' kernel_size,num_output
            cout << "index number: " << i << endl;
            cout << "src_net[" << i << "] type: " << src_net[i].type() << endl;
            cout << "src_net[" << i << "] name: " << src_net[i].name() << endl;
            ConvolutionParameter conv_param = src_net[i].convolution_param();
            num_output=conv_param.num_output();
            if (conv_param.pad_size()>0)
            {
                pad=conv_param.pad(0);
            }
            kernel_size=conv_param.kernel_size(0);
            if (conv_param.stride_size()>0)
            {
                stride=conv_param.stride(0);
            }
            input_size = (input_size + 2 * pad - kernel_size) / stride + 1;
            num_input=num_input/conv_param.group();
        }

        if (i == 1) 
        {
            cout << "num input:" << num_input << endl;
            cout << "num output:"<< num_output << endl;
            load(src_net[i], num_input, num_output, kernel_size);
        }
    }

}

void reload_weight_from_caffe_protobinary(const std::string& caffebinary,int input_param[])
{
    caffe::NetParameter np;

    read_proto_from_binary(caffebinary, &np);
    reload_weight_from_caffe_net(np,input_param);
}