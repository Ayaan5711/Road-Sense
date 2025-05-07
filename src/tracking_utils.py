import supervision as sv

def initialize_tracking(video_info):
    tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=100,
                           match_thresh=0.8, frame_rate=video_info.fps)
    fps_monitor = sv.FPSMonitor()
    return tracker, fps_monitor
