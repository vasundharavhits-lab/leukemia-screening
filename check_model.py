import glob, os, cv2, numpy as np, tensorflow as tf

CLASSES = ["Benign", "Early", "Pre", "Pro"]
m = tf.keras.models.load_model("models/final_mobilenetv2_finetuned.keras", compile=False)
X, y = [], []
for i, c in enumerate(CLASSES):
    for f in sorted(glob.glob(os.path.join(c, "*.jpg"))):
        X.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
        y.append(i)
X = np.stack(X).astype("float32")
y = np.array(y)
p = m.predict(X, verbose=0).argmax(axis=1)
print("images:", len(y), "| accuracy:", round((p == y).mean(), 4), "| errors:", int((p != y).sum()))
files = [f for c in CLASSES for f in sorted(glob.glob(os.path.join(c, "*.jpg")))]
for i in np.where(p != y)[0]:
    print("WRONG:", files[i], "| predicted:", CLASSES[p[i]])

    files = [f for c in CLASSES for f in sorted(glob.glob(os.path.join(c, "*.jpg")))]
for i in np.where(p != y)[0]:
    print("WRONG:", files[i], "| predicted:", CLASSES[p[i]])