sudo apt install libboost-dev

pip install -r requirements.txt

pushd lib/deps/spconv
python setup.py bdist_wheel
pushd dist
pip install *.whl 
popd
rm -rf build dist *.egg*
popd

pushd lib/deps/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
rm -rf build dist *.egg* 
popd

popd
pushd lib/csrc/rroi_align
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/csrc/iou3d
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/csrc/roipool3d
python setup.py install
rm -rf build dist *.egg*
popd

pushd lib/csrc/alignfeature
python setup.py install
rm -rf build dist *.egg*
popd


pushd lib/csrc/correlation
python setup.py install
rm -rf build dist *.egg*
popd

