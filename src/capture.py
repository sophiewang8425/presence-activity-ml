import cv2

def main():
    """Extracts information from webcam"""
    capture=cv2.VideoCapture(0)
    cv2.namedWindow("Webcam")
    # cv2.resizeWindow("Webcam",800,800)
    while True:
        _ , frame= capture.read()
        #create window named webcam and displays the frame
        cv2.imshow("Webcam",frame)
        #manual quit with q key
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        #manual quit by exiting window
        if cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE)<1:
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
