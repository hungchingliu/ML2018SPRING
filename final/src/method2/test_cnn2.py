import os
import sys
from extract import *
from keras.models import load_model

test_pickle = 'pickle/test.npy'
model_path = sys.argv[1]
sample = 'data/sample_submission.csv'
id_name = 'pickle/mfcc_id_name.pickle'
test_dir = sys.argv[2]
out_path = sys.argv[3]

MAX_LEN = 200

if os.path.exists(test_pickle):
    x = np.load(test_pickle)
else:
    x, w2i = read_dir(test_dir)
    x = extract(x)
    np.save(test_pickle, x)

try:
    model = load_model(model_path)
except OSError:
    np.save(out_path, np.load(model_path))
    exit(0)

preds = []
for row in x:
    a = np.zeros([1, 20, MAX_LEN, 1])
    a[0, :, :row.shape[1], 0] = row[:, :MAX_LEN]
    pred = model.predict(a)
    preds.append(pred)

preds = np.array(preds, dtype=np.float64)
# preds = get_sample_order_preds(preds, sample, test_dir)
preds = preds.squeeze()

np.save(out_path, preds)

