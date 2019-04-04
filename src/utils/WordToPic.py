#encoding: utf-8
import os
import pygame

chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
# for codepoint in range(int(start), int(end)):
#     word = unichr(codepoint)
#     font = pygame.font.Font("msyh.ttc", 64)
#     # 当前目录下要有微软雅黑的字体文件msyh.ttc,或者去c:\Windows\Fonts目录下找
#     # 64是生成汉字的字体大小
#     rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
#     pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))

word = "腾讯"
font = pygame.font.Font("msyh.ttc", 64)
rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
