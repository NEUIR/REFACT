# https://www.cnblogs.com/DarkRoger/p/17101082.html

# pip install grpcio -i https://pypi.tuna.tsinghua.edu.cn/simple 
# pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install grpcio-tools -i https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m grpc_tools.protoc -I ./ --python_out=gen/python --grpc_python_out=gen/python agent_service.proto
python3 -m grpc_tools.protoc -I ./ --python_out=gen/python --grpc_python_out=gen/python general_servlet_grpc.proto
python3 -m grpc_tools.protoc -I ./ --python_out=gen/python --grpc_python_out=gen/python helloworld.proto
python3 -m grpc_tools.protoc -I ./ --python_out=gen/python --grpc_python_out=gen/python router.proto
