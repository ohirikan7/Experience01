-n AdafaceTest仮想環境による実行
pytorch を最新バージョン、GPU3の環境に合わせたcudaバージョン

conda create -n AdafaceTest python=3.12
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-image matplotlib pandas scikit-learn
pip install pytorch-lightning==1.8.6
pip install tqdm bcolz-zipline prettytable menpo
pip install mxnet opencv-python
