import cv2
import time
from presence_model import detect_person


def main():
    capture=cv2.VideoCapture(0)
    while True:
        _,frame=capture.read()
        presence=detect_person(frame)

        if presence:
            label="HUMAN PRESENT"
            #green
            color=(0, 255,0)
        else:
            label="NO HUMAN"
            #red
            color=(0,0,255)
        cv2.putText(frame,label,(30,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        
        if presence:
            cv2.imwrite(f"data/raw/present/{time.time()}.jpg",frame)
        else:
            cv2.imwrite(f"data/raw/empty/{time.time()}.jpg",frame)
        
        cv2.imshow("Presence Detector",frame)
        
        #manual quit with q key
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        #manual quit by exiting window
        if cv2.getWindowProperty("Presence Detector", cv2.WND_PROP_VISIBLE)<1:
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()