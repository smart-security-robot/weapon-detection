# weapon-detection
weapon detection inference

# Weight

'''
https://drive.google.com/file/d/1Wj9NXaf8prIcem-qyfL9Ca-BBW5xILl5/view?usp=sharing
'''
sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --device /dev/video0:/dev/video0 -p 8000:8000 weapon-detect
