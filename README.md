# face-upright-tracking
Tracking face is at upright position, could be used to check attention-or-not or other use cases.

Installation of ffmpeg server:
pip install ffmpeg-python

run app and send output video to ffmpeg server for video streaming:
python app.py | ffmpeg -f rawvideo -pixel_format bgr2 -framerate 24 -video_size 1280x720 -i - http://0.0.0.0:3004/fac.ffm
