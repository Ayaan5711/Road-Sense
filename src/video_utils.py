from vidgear.gears import CamGear

def extract_video_url(source: str, resolution='1280x720') -> str:
    stream = CamGear(source=source, stream_mode=True, logging=True).start()
    for fmt in stream.ytv_metadata["formats"]:
        if fmt["resolution"] == resolution:
            return fmt["url"]
    raise ValueError(f"Resolution {resolution} not found.")

def video_manifest_extractor(source):
        """
        Function to extract metadata from a YouTube video source
          and find the desired resolution URL.

        Parameters:
        source (str): Video source URL (ex. "<https://youtu.be/bvetuLwJIkA>")

        Returns:
        str: Desired resolution video URL
        """
        stream = CamGear(source=source, stream_mode=True, logging=True,
                         time_delay=0).start()
        video_metadata = stream.ytv_metadata

        

        resolutions = [format["resolution"] for format in video_metadata["formats"]]
        for res in resolutions:
            print(res)

        resolution_desiree = '1280x720'
        for format in video_metadata["formats"]:
            if format["resolution"] == resolution_desiree:
                VIDEO = format["url"]
                return VIDEO
