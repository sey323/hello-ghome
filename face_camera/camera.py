import sys
import cv2
import detector

def camera_facedetect(save_path):
    # カメラの設定
    cap = cv2.VideoCapture(0)
    end_flag, frame = cap.read()

    while(True):
        if cv2.waitKey(1) == 27:
            break

        # 顔の検出と保存
        image = frame
        face_list = detector.face_detect( image )
        detector.save_faceImage( image , face_list , base = 64 ,  save_path = save_path)

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def main(save_path):
    camera_facedetect(save_path)


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    if(argc != 2):
    	print("引数を指定して実行してください。")
    	quit()

    save_path = args[1]
    main(save_path)
