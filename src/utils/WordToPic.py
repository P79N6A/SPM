#encoding: utf-8
import os
import pygame
from PIL import Image
import json

if __name__ == "__main__":
    data = "multi_data7_tech"
    data_path = 
    with open()

chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()



word = "腾讯"
font = pygame.font.Font("data/msyh.ttc", 64)
rtext = font.render(word, True, (0, 0, 0))
pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))


arr = ['p1.png', 'p2.png', 'p3.png', 'p4.png']
toImage = Image.new('RGBA',(400,400))
for i in range(4):
    fromImge = Image.open(arr[i])
    # loc = ((i % 2) * 200, (int(i/2) * 200))
    loc = ((int(i/2) * 200), (i % 2) * 200)
    print(loc)
    toImage.paste(fromImge, loc)

toImage.save('merged.png')