import cv2
from lane_detection import detect_lane_lines


def process_video(input_path: str, output_path: str | None = None):
    """
    Apply lane detection to every frame of the input video.
    Optionally save the result to 'output_path'.
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    # Optional: prepare video writer if we want to save the result
    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = detect_lane_lines(frame)

        cv2.imshow("Lane Detection - Video", processed)

        if writer is not None:
            writer.write(processed)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video_path = "data/test_video.mp4"  # place your dashcam video here
    output_video_path = "data/output_lane_detection.mp4"  # or None if you don't want to save

    process_video(input_video_path, output_video_path)
