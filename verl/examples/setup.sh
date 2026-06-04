rm /etc/xdg/pip/pip.conf
rm /etc/pip.conf
rm /root/.pip/pip.conf
rm /root/.config/pip/pip.conf

if [ $CLUSTER == "ningxia_h100" ]; then
   pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
elif [ $CLUSTER == "kunming_train" ]; then
   pip config set global.index-url https://nexus.xn-01.zetyun.cn/repository/pypi/simple
else
   #pip config set global.index-url https://pypi.hs1.paratera.com/root/pypi/+simple
   pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
fi

pip install optree -U
pip install vllm==0.8.5.post1
pip install peft accelerate -U
pip install /user/jinzhensheng/tools/block_sparse_attn-0.0.1-cp310-cp310-linux_x86_64.whl
pip install /user/jinzhensheng/tools/nsa-0.0.0-cp310-cp310-linux_x86_64.whl
pip install timeout_decorator polars seaborn
pip install google-cloud-aiplatform latex2sympy2 pylatexenc sentence_transformers tabulate vertexai
pip install math-verify antlr4-python3-runtime -U
pip install /user/jinzhensheng/tools/omegaconf-2.4.0.dev3-py3-none-any.whl
pip install /user/jinzhensheng/tools/hydra_core-1.4.0.dev1-py3-none-any.whl
pip install grpcio grpcio-tools protobuf loguru
pip install tokenizers==0.21.0
pip install swanlab
pip install -U --no-cache-dir "pyarrow>=16" "pandas>=2.2.2"
pip install transformers==4.57 
cp ./minicpm.py /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/minicpm.py