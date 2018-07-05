cd method2/
test_dir=../${1}

for i in {0..10};
do
    echo predict mfcc ${i}
    python3 test_cnn2.py model/mfcc${i}.mdn ${test_dir} ../predict/mfcc${i}.npy
done

for i in {7..9}
do
    echo predict 1d-conv ${i}
    python3 test_raw.py ${test_dir} model/1d_conv${i}.mdn ../predict/1d_conv${i}.npy
done

cd ../method_1/

echo method_1
test_dir=../${1}
git clone https://gitlab.com/harry1003/ml_final.git
python3 generate_answer.py ${test_dir}

cd ..
echo ensemble
python3 ensemble.py ${2}