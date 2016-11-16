import av
import os

container = av.open('big_buck_bunny_720p_5mb.mp4')
video = next(s for s in container.streams if s.type == 'video')

for packet in container.demux(video):
    for frame in packet.decode():
        frame.to_image().save('frames/frame-%04d.jpg' % frame.index)


# import cv2
# vidcap = cv2.VideoCapture('/home/hii/Projects/PiQual/video/big_buck_bunny_720p_5mb.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#   count += 1
