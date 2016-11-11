import cv2
vidcap = cv2.VideoCapture('destiny_animation_short.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
