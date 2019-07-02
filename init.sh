sudo apt install libboost-dev

pip install -r requirements.txt


pushd lib/csrc/alignfeature
python setup.py install
rm -rf build dist *.egg*
popd


pushd lib/csrc/correlation
python setup.py install
rm -rf build dist *.egg*
popd

