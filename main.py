import cv2 as cv


def image_proc():
    img = cv.imread('img_test.jpg')
    h, w = img.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv.getRotationMatrix2D((cX, cY), 90, 1.0)
    rotated = cv.warpAffine(img, M, (w, h))
    cv.imshow('rotated', rotated)


def video_proc():
    cap = cv.VideoCapture('cam_video.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21,21), 0)
        ret, thresh = cv.threshold(gray, 105, 255, cv.THRESH_BINARY_INV)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv.contourArea)
            x1,y1,w,h = cv.boundingRect(c)
            cv.rectangle(frame, (x1,y1), (x1+w, y1+h), (255,105,0),2)
            x2 = x1+w
            y2 = y1+h

            width = x2 - x1
            half_width = width // 2
            height = y2 - y1
            start_point = x1,y1+half_width
            end_point = x2,y1+half_width
            color = (255,0 , 255)

            # Горизонтальная линия
            cv.line(frame,start_point,end_point,color,3)

            #Вертикальная линия
            start_point = x1+ half_width, y1
            end_point = x1+ half_width, y2
            cv.line(frame, start_point, end_point, color,3)

        cv.imshow('frame', frame)
        cv.waitKey(0)


if __name__ == '__main__':
    video_proc()

cv.waitKey(0)
cv.destroyAllWindows()
