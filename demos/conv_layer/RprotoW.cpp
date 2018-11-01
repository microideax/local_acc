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
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "layer_factory_impl.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#define CNN_OPEN_BINARY(filename) open(filename, O_RDONLY)
#define CNN_OPEN_TXT(filename) open(filename, O_RDONLY)

using namespace caffe;
using namespace std;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

void load(const caffe::LayerParameter& src,int num_input,int num_output,int kernel_size) {
    int src_idx = 0;
    ofstream out;
    out.open("net_weights.txt",ios::app);
    cout<<"weights: "<<endl;
    //load weight
    for (int o = 0; o < num_output; o++) {
        for (int i = 0; i < num_input; i++) {
            for (int x = 0; x < kernel_size * kernel_size; x++) {
                cout<<src.blobs(0).data(src_idx++)<<" ";
            }
        }
    }
    cout<<""<<endl;
    if(src.convolution_param().bias_term()==false){

    }else{
        cout<<"bias: "<<endl;
        //load bias
        for (int o = 0; o < num_output; o++) {
            cout<<src.blobs(1).data(o)<<" ";
        }
        cout<<""<<endl;
    }
    out.close();
}

void read_proto_from_text(const std::string& prototxt,
                          google::protobuf::Message *message) {
    int fd = CNN_OPEN_TXT(prototxt.c_str());
    if (fd == -1) {
        cout<<"file not found: "<<prototxt<<endl;
    }

    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);

    if (!google::protobuf::TextFormat::Parse(&input, message)) {
        cout<<"failed to parse"<<endl;
    }
}

void create_net_from_caffe_net(const caffe::NetParameter& layer,int input_param[])
{
    cout << "create net from caffe net" << endl;
    caffe_layer_vector src_net(layer);
    if (layer.input_shape_size() > 0) {
        // input_shape is deprecated in Caffe
        // blob dimensions are ordered by number N x channel K x height H x width W
        input_param[0]  = static_cast<int>(layer.input_shape(0).dim(1));
        cout << "input_param[0]: " << input_param[0] << endl;
        input_param[1] = static_cast<int>(layer.input_shape(0).dim(2));
        cout << "input_param[1]: " << input_param[1] << endl;
        //int width  = static_cast<int>(layer.input_shape(0).dim(3));
        //cout<<"depth:********************"<<depth<<endl;
    }else if (layer.layer(0).has_input_param()) {
        cout << "input has layer param " << endl;
        // blob dimensions are ordered by number N x channel K x height H x width W
        input_param[0] = static_cast<int>(layer.layer(0).input_param().shape(0).dim(1));
        input_param[1] = static_cast<int>(layer.layer(0).input_param().shape(0).dim(2));
        cout << "input_param[0]: " << input_param[0] << endl;
        cout << "input_param[1]: " << input_param[1] << endl;
        //int kernel_h = static_cast<int>(layer.layer(0).input_param().shape(0).dim(2));
        //int kernel_w = static_cast<int>(layer.layer(0).input_param().shape(0).dim(3));
        //cout<<"width:********************"<<kernel_w<<endl;
        //return input_param;
    }//else if (layer.input_dim_size() > 0){
    //   input_param[0] = static_cast<int>(layer.input_dim(1));
    //   input_param[1] = static_cast<int>(layer.input_dim(2));
    //   cout<<"no input shape: "<<endl;
    //}
}

void create_net_from_caffe_prototxt(const std::string& caffeprototxt,int input_param[])
{
    caffe::NetParameter np;
    read_proto_from_text(caffeprototxt, &np);
    cout <<"net_name from prototxt: "<<np.name()<<endl;
    create_net_from_caffe_net(np,input_param);
}

void read_proto_from_binary(const std::string& protobinary,
                            google::protobuf::Message *message) {
    int fd = CNN_OPEN_BINARY(protobinary.c_str());
    google::protobuf::io::FileInputStream rawstr(fd);
    google::protobuf::io::CodedInputStream codedstr(&rawstr);

    rawstr.SetCloseOnDelete(true);
    codedstr.SetTotalBytesLimit(std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::max() / 2);

    if (!message->ParseFromCodedStream(&codedstr)) {
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
    for (int i = 0; i < src_net.size(); i++) {
        int pad=0;
        int kernel_size=0;
        int stride=1;

        /*
        // name，type，kernel size，pad，stride
        if(src_net[i].type()=="Convolution"||src_net[i].type()=="ConvolutionRistretto"){//get conv_layers' kernel_size,num_output
            ConvolutionParameter conv_param = src_net[i].convolution_param();
            num_output=conv_param.num_output();
            if (conv_param.pad_size()>0){
                pad=conv_param.pad(0);
            }
            //google::protobuf::RepeatedField<string> repeated_field;
            //repeated_field.size();
            kernel_size=conv_param.kernel_size(0);
            if (conv_param.stride_size()>0){
                stride=conv_param.stride(0);
            }
            input_size = (input_size + 2 * pad - kernel_size) / stride + 1;
            num_input=num_input/conv_param.group();
        }else if(src_net[i].type()=="InnerProduct"){//get fc_layers' kernel_size,num_output
            InnerProductParameter inner_product_param = src_net[i].inner_product_param();
            kernel_size=input_size;
            num_output=inner_product_param.num_output();
            input_size=1;
        }else if(src_net[i].type()=="Pooling"){//get pooling_layers' kernel_size,num_output
            PoolingParameter pooling_param = src_net[i].pooling_param();
            pad=pooling_param.pad();
            kernel_size=pooling_param.kernel_size();
            stride=pooling_param.stride();
            input_size = static_cast<int>(ceil(static_cast<float>(input_size + 2 * pad - kernel_size) / stride)) + 1;
        }

        if(src_net[i].type()=="Convolution"||src_net[i].type()=="InnerProduct"){
            load(src_net[i],num_input,num_output,kernel_size);
        }
        if(src_net[i].type()=="Convolution"||src_net[i].type()=="InnerProduct"||src_net[i].type()=="Pooling"){
            num_input=num_output;//set each layer's num_input equals to the last layer's num_output
        }
        */

        if(src_net[i].type()=="Convolution"||src_net[i].type()=="ConvolutionRistretto"){//get conv_layers' kernel_size,num_output
            cout << "index number: " << i << endl;
            cout << "src_net[" << i << "] type: " << src_net[i].type() << endl;
            cout << "src_net[" << i << "] name: " << src_net[i].name() << endl;
            ConvolutionParameter conv_param = src_net[i].convolution_param();
            num_output=conv_param.num_output();
            if (conv_param.pad_size()>0){
                pad=conv_param.pad(0);
            }
            //google::protobuf::RepeatedField<string> repeated_field;
            //repeated_field.size();
            kernel_size=conv_param.kernel_size(0);
            if (conv_param.stride_size()>0){
                stride=conv_param.stride(0);
            }
            input_size = (input_size + 2 * pad - kernel_size) / stride + 1;
            num_input=num_input/conv_param.group();
        }
        if (i ==1 ) {
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

int main(int argn, char* argv[])
{
    string prototxt = argv[2];
    string caffemodel = argv[1];
    cout << "Read prototxt file from: " << prototxt << endl;
    cout << "Read caffemodel file from: " << caffemodel << endl;

    NetParameter proto;
    int input_param[] = {0, 0};

//	ReadProtoFromBinaryFile(argv[1], &proto);
    cout << "proto layers size: " << proto.layers_size() << endl;
    cout << "proto bytesize: " << proto.ByteSize() << endl;

    create_net_from_caffe_prototxt(argv[2], input_param);
    reload_weight_from_caffe_protobinary(argv[1], input_param);



//    cout << proto.layer << endl;
	//WriteProtoToTextFile(proto, argv[2]);
	return 0;
}


