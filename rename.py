import os
import cv2

# 포켓몬 폴더 경로
file_path = 'D:\\Gongmo\\It_worked\\img\\9\\Pikachu_pp\\'
file_names = os.listdir(file_path)

os.chdir(file_path)

def resize_save():
    for i, name in enumerate(file_names):
        src = cv2.imread(file_path + name, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(src, dsize=(200, 200), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(name, resized_img)

def rename():
    for i, name in enumerate(file_names):
        new_name = str(i) + '.jpg'
        os.renames(name, new_name)

if __name__ == "__main__":
    resize_save()
    rename()