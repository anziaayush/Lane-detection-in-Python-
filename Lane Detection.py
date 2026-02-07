import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    kernel_size must be an odd number (e.g., 3, 5, 7).
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_canny(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Apply Canny edge detection.
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def region_of_interest(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Apply an image mask that keeps only the region defined by 'vertices'.

    Parameters:
        image: Single-channel or 3-channel image.
        vertices: Polygon describing the region of interest.
    """
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        # Color image
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # Grayscale image
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def make_line_points(y1: int, y2: int, line: np.ndarray) -> tuple[int, int, int, int]:
    """
    Convert a line in slope/intercept form into pixel coordinates.
    'line' is [slope, intercept].
    """
    slope, intercept = line
    if slope == 0:
        # Avoid division by zero
        slope = 1e-6

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2


def average_slope_intercept(image: np.ndarray, lines: np.ndarray):
    """
    Separate left and right lines, average them, and return their coordinates.
    """
    left_lines = []   # (slope, intercept)
    right_lines = []  # (slope, intercept)

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                # Skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Filter out nearly horizontal lines
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    left_line = np.mean(left_lines, axis=0) if left_lines else None
    right_line = np.mean(right_lines, axis=0) if right_lines else None

    height = image.shape[0]
    y1 = height                 # bottom of the image
    y2 = int(height * 0.6)      # a bit lower than the middle

    left_points = make_line_points(y1, y2, left_line) if left_line is not None else None
    right_points = make_line_points(y1, y2, right_line) if right_line is not None else None

    return left_points, right_points


def draw_lines(image: np.ndarray, lines, color=(0, 255, 0), thickness: int = 10) -> np.ndarray:
    """
    Draw lines on the image.
    'lines' should be a list of (x1, y1, x2, y2) tuples or None.
    """
    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            if line is None:
                continue
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    # Blend original image and line image
    combined = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
    return combined


def detect_lane_lines(frame: np.ndarray) -> np.ndarray:
    """
    Complete lane detection pipeline for a single frame.
    Returns the frame with lane lines drawn on top.
    """
    # 1. Preprocessing: grayscale and blur
    gray = to_grayscale(frame)
    blur = apply_gaussian_blur(gray, kernel_size=5)

    # 2. Canny edge detection
    edges = apply_canny(blur, low_threshold=50, high_threshold=150)

    # 3. Region of Interest (ROI)
    height, width = frame.shape[:2]
    # Define a trapezoid region focusing on the lane area
    roi_vertices = np.array([
        (int(0.1 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ], dtype=np.int32)

    masked_edges = region_of_interest(edges, roi_vertices)

    # 4. Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # 5. Average and extrapolate left and right lane lines
    left_line, right_line = average_slope_intercept(frame, lines)

    # 6. Draw lines and return result
    result = draw_lines(frame, [left_line, right_line])
    return result


if __name__ == "__main__":
    """
    Simple manual test: load an image, run lane detection, and show the result.
    Adjust 'test_image_path' to your own image file if you want to test this.
    """
    test_image_path = "data/test_image.jpg"  # optional test image

    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Could not load test image at '{test_image_path}'.")
    else:
        output = detect_lane_lines(image)
        cv2.imshow("Lane Detection - Test Image", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
