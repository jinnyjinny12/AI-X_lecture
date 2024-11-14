import cv2
import time

class Button:
    def __init__(self, x, y, w, h, text, color=(0, 255, 0), active_color=(0, 0, 255)):
        """
        Initialize the button with position, size, text, and colors.
        :param x: Top-left x-coordinate
        :param y: Top-left y-coordinate
        :param w: Width of the button
        :param h: Height of the button
        :param text: Text to display on the button
        :param color: Normal button color
        :param active_color: Button color when active
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.active_color = active_color
        self.start_time = None
        self.active = False  # Track if the button is active

    def draw(self, img):
        """
        Draw the button on the image.
        :param img: Image on which to draw
        """
        # Choose color based on the button's active state
        button_color = self.active_color if self.active else self.color

        # Draw filled rectangle if active, else outline
        thickness = -1 if self.active else 2
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), button_color, thickness)

        # Add button text
        text_color = (255, 255, 255) if self.active else self.color
        cv2.putText(img, self.text, (self.x + 10, self.y + self.h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    def is_hovering(self, point):
        """
        Check if the given point is within the button.
        :param point: (x, y) coordinates of the point
        :return: True if hovering, False otherwise
        """
        x, y = point
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def detect_action(self, hovering, duration=1):
        """
        Detect if the user has hovered over the button for a specific duration.
        :param hovering: True if the point is hovering over the button
        :param duration: Duration to detect the action
        :return: True if action is detected, False otherwise
        """
        if hovering:
            if self.start_time is None:
                self.start_time = time.time()
            elif time.time() - self.start_time >= duration:
                self.active = True  # Set button as active
                return True
        else:
            self.start_time = None
            self.active = False  # Reset button state if not hovering
        return False