import cv2
from pathlib import Path
import shutil
from itertools import chain

RAW = Path("data/raw")
OUT = Path("data/cleaned")

OUT_PRESENT = OUT / "present"
OUT_EMPTY   = OUT / "empty"
OUT_PRESENT.mkdir(parents=True, exist_ok=True)
OUT_EMPTY.mkdir(parents=True, exist_ok=True)

# collect everything YOLO produced
files = list(chain(
    (RAW / "present").glob("*.jpg"),
    (RAW / "empty").glob("*.jpg"),
))

print("\nControls:")
print("  p = present (human visible)")
print("  e = empty   (no human)")
print("  s = skip    (drop from dataset)")
print("  q = quit\n")

for img_path in files:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    cv2.imshow("Clean Dataset", img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('p'):
            shutil.copy(img_path, OUT_PRESENT / img_path.name)
            break

        elif key == ord('e'):
            shutil.copy(img_path, OUT_EMPTY / img_path.name)
            break

        elif key == ord('s'):
            break

        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
