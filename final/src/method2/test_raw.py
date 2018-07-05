import sys
from extract import *
from keras.models import load_model

folder = sys.argv[1]
x, wave2ind = read_dir(folder)
files = get_fnames(folder)

d = np.zeros([len(files), 88200])
for i in range(len(x)):
    d[i][:len(x[i])] = x[i][:88200]
d = np.expand_dims(d, axis=-1)

model = load_model(sys.argv[2])
preds = model.predict(d)

np.save(sys.argv[3], preds)

