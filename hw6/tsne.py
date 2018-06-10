from keras.models import load_model
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

def read_cat():
    dict = {'Animation': 2, 'Adventure': 0, 'Comedy': 1, 'Action': 0,
            'Drama': 1, 'Thriller': 5, 'Crime': 4, 'Romance': 1,
            "Children's": 2, 'Documentary': 4, 'Sci-Fi': 0, 'Horror': 5,
            'Western': 0, 'Mystery': 0, 'Film-Noir': 4, 'War': 4, 'Musical': 1,
            'Fantasy': 0}
    with open("dataset/movies.csv", "r", encoding='latin-1') as file:
        reader = csv.reader(file)
        cat = []
        id = 1
        flag = 0
        for line in reader:
            if len(line) > 1:
                line = [line[0] + line[1]]
            #print(line)
            if flag == 0:
                flag = 1
                continue
            line = line[0].split("::")
            id_now = line[0]
            while(id < int(id_now)):
                cat.append(3)
                id += 1
            #print(line)
            #data.append(line[0])
            if len(line) > 2:
                data_cat = line[2].split("|")[0]
                cat.append(dict.get(data_cat, 3))
            else:
                cat.append(3)
            #print(data)
            id += 1
    return cat
def main():
    cat = read_cat()
    model = load_model("MF.h5")

    #get embedding

    movie_emb = np.array(model.layers[3].get_weights()).squeeze()
    cat = np.array(cat, dtype=np.float64).squeeze()
    print(cat.shape)
    """vis_data = TSNE(n_components=2).fit_transform(movie_emb)
    np.save("tsne.npy", vis_data)"""
    vis_data = np.load("tsne.npy")
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    plt.figure(figsize=(7, 7), dpi=200)
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(vis_x, vis_y, marker='.',s=20, c=cat, cmap=cm)
    plt.colorbar(sc)
    plt.savefig("tsne.png")

if __name__ == "__main__":
    main()