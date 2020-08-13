import os

# 포켓몬 폴더 경로
file_path = 'D:\\Gongmo\\It_worked\\img\\8\\Purin\\'
file_names = os.listdir(file_path)

os.chdir(file_path)
i = 1
for name in file_names:
    new_name = str(i) + '.jpg'
    os.rename(name, new_name)
    i += 1