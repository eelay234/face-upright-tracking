# face-upright-tracking
Tracking face at upright position, it could be used to check attention-or-not or other use cases.

**Input** is video file or camera input that has person face(s)

**Output** is the streaming video showing bounding box and facial landmarks of faces and whether the face is at upright position.

**Setup and installation:**
- Download and install Intel Openvino(R) toolkits.
- Installation of ffmpeg server

**Run app and send output video to ffmpeg server for video streaming:**

python app.py | ffmpeg -f rawvideo -pixel_format bgr2 -framerate 24 -video_size 1280x720 -i - http://0.0.0.0:3004/fac.ffm
