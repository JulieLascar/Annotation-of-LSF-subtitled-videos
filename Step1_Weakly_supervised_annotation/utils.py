import cv2


def frames2video(video_path: str, path_out: str, L_frames, fps=25):
    """
    Convert a list of frames to video and saves it in a directory.

    Args :
        video_path (str): path to the video
        path_out (str): path to save the video
        L_frames (list): list of frames

    """
    if isinstance(L_frames, set):
        L_frames = list(L_frames)

    L_frames.sort()

    # cuting videos
    cap = cv2.VideoCapture(video_path)
    L_Videos = []

    for k in range(7 + L_frames[0]):
        _, frame = cap.read()

    for k in range(len(L_frames)):
        _, frame = cap.read()
        L_Videos.append(frame)

    # creating videos
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define the codec

    try:
        height, width, _ = L_Videos[0].shape
        video = cv2.VideoWriter(path_out, fourcc, fps, (width, height))
        for k in range(len(L_Videos)):
            video.write(L_Videos[k])
        video.release()
        cap.release()

    except:
        print("problem with  :", video_path)

    cv2.destroyAllWindows()


def change_video_pfs(video_path, path_out, fps=10):
    """
    Change the fps of a video.

    Args :
        video_path (str): path to the video
        path_out (str): path to save the video
        fps (int): new fps
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path_out, fourcc, fps, (width, height), True)

    ret = True
    while ret:
        ret, frame = cap.read()
        video.write(frame)

    print("done")
    cap.release()
    video.release()
    cv2.destroyAllWindows()
