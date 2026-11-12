import glob, cv2
from src.model import check_image
ok = n = 0
for c in ["Benign", "Early", "Pre", "Pro"]:
    for f in glob.glob(c + "/*.jpg"):
        n += 1
        ok += check_image(cv2.imread(f))[0] is not None
print(ok, "of", n, "pass the input check")
