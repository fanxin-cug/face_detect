#导入第三方库
import cv2
import dlib
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("/home/fanxin/My_FaceShape_LandMark.dat")#自己训练好的特征检测器
img = cv2.imread("/home/fanxin/Downloads/Image/2.jpg")
faces = detector(img,1)
if (len(faces) > 0):
    X_list=list()
    Y_list=list()
    section=[18,6,6,7,7,8,10,8]#下巴、左眉毛、右眉毛、左眼、右眼、鼻子、上嘴唇、下嘴唇
    for d in enumerate(faces):
        cv2.rectangle(img,(d.left(),d.top()),(d.right(),d.bottom()),(0,0,255),5)
        shape = landmark_predictor(img,d)
        #for i in range(70):
            #cv2.circle(img, (shape.part(i).x, shape.part(i).y),8,(0,255,0), -1, 8)
        ans=0
        for i in range(len(section)):
            for j in range(section[i]):
                X_list.append(shape.part(ans+j).x)
                Y_list.append(-shape.part(ans+j).y)
            plt.plot(X_list,Y_list)
            X_list.clear()
            Y_list.clear()
            ans+=section[i]
plt.show()
#cv2.imshow("face",img)
#cv2.imwrite("/home/fanxin/Pictures/2.jpg",img)
#cv2.waitKey(0)
