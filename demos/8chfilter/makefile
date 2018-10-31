CC = g++
TARGET = filter
LIBS = `pkg-config opencv --libs`
INC = -I../../ACC_API/include
INC += -I.`pkg-config opencv --cflags`
SRCS = filter.cpp ../../ACC_API/src/layer.cpp  ../../ACC_API/src/acc_ctrl.cpp
OBJS = $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

%.o: %.cpp 
	$(CC) -c $< -o $@ $(INC)

clean:
	rm -f *.o ../../ACC_API/src/*.o $(TARGET)