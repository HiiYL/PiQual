import av
import os

container = av.open('destiny_animation_short.mp4')
video = next(s for s in container.streams if s.type == 'video')

for packet in container.demux(video):
    for frame in packet.decode():
        frame.to_image().save('frames/frame-%04d.jpg' % frame.index)


# import cv2
# vidcap = cv2.VideoCapture('destiny_animation_short.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
#   count += 1
