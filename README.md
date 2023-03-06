环境安装：
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install spacy==2.1.0
pip install en_vectors_web_lg-2.1.0.tar.gz
pip uninstall opencv-python-headless
pip install opencv-python==4.5.5.64
pip install opencv-contrib-python
pip install -r requirements.txt
