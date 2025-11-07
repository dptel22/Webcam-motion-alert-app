import glob
import os
import cv2
import time
from emailing import send_email
from threading import Thread

video = cv2.VideoCapture(0)
time.sleep(1)

status_list = []
count = 1
image_with_object = None

def clean_folder():
    print("Cleaning the folder")
    images = glob.glob("images/*.png")
    for image in images:
        try:
            os.remove(image)
        except Exception as e:
            print(f"Failed to remove {image}: {e}")
    print("Cleaned the folder")

first_frame = None
while True:
    status = 0
    check, frame = video.read()

    if not check or frame is None:
        print("Warning: failed to read frame from camera. Skipping iteration.")
        time.sleep(0.1)
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_gau_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_gau_frame

    delta_frame = cv2.absdiff(first_frame, gray_gau_frame)

    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    div_frame = cv2.dilate(thresh_frame, None, iterations=2)
    # cv2.imshow("My Video", div_frame)

    contours, _ = cv2.findContours(div_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if rectangle.any():
            status = 1
            # save frame
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            if all_images:
                image_with_object = all_images[index]
            else:
                image_with_object = None

    status_list.append(status)
    status_list = status_list[-2:]


    if len(status_list) == 2 and status_list[0] == 1 and status_list[1] == 0:
        if image_with_object is None:
            print("Motion detected then stopped but no image was saved. Skipping email/cleanup.")
        else:
            print(f"Triggering email thread for: {image_with_object}")

            email_thread = Thread(target=send_email, args=(image_with_object,))
            email_thread.daemon = True

            clean_thread = Thread(target=clean_folder)
            clean_thread.daemon = True

            email_thread.start()
            clean_thread.start()

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()


print("Program terminated.")
