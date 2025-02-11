import cv2
import numpy as np
import time
class VideoProcessor:
    def __init__(self, video_path, target_resolution=(640, 360), fps_interval=10):
        self.video_path = video_path
        self.target_resolution = target_resolution
        self.fps_interval = fps_interval
        self.color_ranges = [
            (np.array([0, 80, 40]), np.array([10, 255, 255])),  # Açık kırmızı ve turuncumsu kırmızı
            (np.array([160, 80, 40]), np.array([180, 255, 255])) # Koyu kırmızı tonları
        ]
        self.cap = self.initialize_video_capture()
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
    def initialize_video_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("Video file could not be opened!")
        return cap
    def preprocess_frame(self, frame):
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        return hsv_image
    def create_color_masks(self, hsv_image):
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges:
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_image, lower, upper))
        return combined_mask
    def apply_morphological_operations(self, mask, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask
    def filter_roi(self, mask, frame_shape):
        h, w = frame_shape[:2]
        x_start, y_start = int(w * 0.25), int(h * 0.25)
        x_end, y_end = int(w * 0.75), int(h * 0.75)
        roi_mask = np.zeros_like(mask)
        roi_mask[y_start:y_end, x_start:x_end] = mask[y_start:y_end, x_start:x_end]
        return roi_mask
    def calculate_colored_percentage(self, mask):
        colored_area = np.count_nonzero(mask)
        total_area = mask.size
        return (colored_area / total_area) * 100
    def annotate_frame(self, frame, classification, colored_percentage):
        cv2.putText(frame, f"Classification: {classification}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Percentage: {colored_percentage:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    def draw_bounding_boxes(self, frame, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum alan filtresi
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    def calculate_fps(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream")
                break
            frame = cv2.resize(frame, self.target_resolution)
            self.frame_count += 1
            hsv_image = self.preprocess_frame(frame)
            combined_mask = self.create_color_masks(hsv_image)
            combined_mask = self.apply_morphological_operations(combined_mask)
            roi_mask = self.filter_roi(combined_mask, frame.shape)
            highlighted_image = frame.copy()
            highlighted_image[roi_mask > 0] = [0, 0, 255]  # Kırmızı renk
            self.draw_bounding_boxes(highlighted_image, roi_mask)
            colored_percentage = self.calculate_colored_percentage(roi_mask)
            classification = "dolu" if colored_percentage > 0 else "boş"
            if self.frame_count % self.fps_interval == 0:
                self.calculate_fps()
            self.annotate_frame(highlighted_image, classification, colored_percentage)
            cv2.imshow("Original Feed", frame)
            cv2.imshow("Highlighted Areas", highlighted_image)
            print(f"Frame {self.frame_count}: Colored Area: {colored_percentage:.2f}%, FPS: {self.fps:.2f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    video_path = "rtsp://admin:oms12345@192.168.1.64:554/stream"
    processor = VideoProcessor(video_path)
    processor.process_video()
    
    