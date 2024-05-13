import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CSRNet
import torch
from torch.autograd import Variable
import cv2
import numpy as np
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error

from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import datetime
import os.path
import time
import PySimpleGUI as sg
import cv2
import PySimpleGUI as psg
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images
def now_to_date(format_string="%Y-%m-%d %H:%M:%S"):

    time_stamp = int(time.time())

    time_array = time.localtime(time_stamp)

    str_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)

    return str_date

def count():
    time_stamp = int(time.time())

    time_array = time.localtime(time_stamp)

    str_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)

    img_folder='1/12'

    img_paths=[]

    for img_path in glob.glob(os.path.join( img_folder, '*.jpg')):
        img_paths.append(img_path)

    model = CSRNet()

    model = model.cuda()

    checkpoint = torch.load('model_best.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    pred= []
    gt = []

    for i in range(len(img_paths)):
        img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
        img = img.unsqueeze(0)
        h,w = img.shape[2:4]
        h_d = h//2
        w_d = w//2
        img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
        img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
        img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
        img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
        density_1 = model(img_1).data.cpu().numpy()
        density_2 = model(img_2).data.cpu().numpy()
        density_3 = model(img_3).data.cpu().numpy()
        density_4 = model(img_4).data.cpu().numpy()

        # 将上部两张图片进行拼接,...为表示省略表示后面参数都全选
        up_map=np.concatenate((density_1[0,0,...],density_2[0,0,...]),axis=1)
        down_map=np.concatenate((density_3[0,0,...],density_4[0,0,...]),axis=1)
        # 将上下部合成一张完成图片
        final_map=np.concatenate((up_map,down_map),axis=0)
        plt.imshow(final_map,cmap=cm.jet) # 展示图片
        #plt.savefig("/img/"+i+".png")
        plt.show()
        sg.cprint(int(final_map.sum()) ,"\t","\t","\t","\t",str_date)# 直接输出图像预测的人数
        i=1


'''
def get_img_from_camera_net(url):
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
            # 不实时显示监控画面
        #cv2.imshow("frame", frame)
        file = "H:\ma\CSRNet-pytorch-master/1/"+ "{}.jpg".format(datetime.datetime.now().strftime("1"))  # 以日期来命名文件并指定文件夹
        file1 = "H:\ma\CSRNet-pytorch-master/1/"+ "{}.png".format(datetime.datetime.now().strftime("1"))
        #if (int(time.strftime("%S", time.localtime())) % 5 == 0):  
        cv2.imwrite(file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 无损输出图片 但是图片质量仍不是很高
        cv2.imwrite(file1, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 9])
        file1 =cv2.imread('1/1.PNG')
        height,width = file1.shape[:2]
        file1 = cv2.resize(file1,(width//2,height//2),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('1/12/1.PNG',file1)
        print(file)
        if os.path.isfile(file) == True:  # 按q退出
            break
    cv2.destroyAllWindows()  # 全屏显示
'''

def cut(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(cnts)
    r = max(w/2, h/2)
    # 提取有效区域
    img_valid = img[y:y+h, x:x+w]
    return img_valid, int(r)

# 鱼眼矫正
def undistort(src,r):
    # r： 半径， R: 直径
    R = 2*r
    # Pi: 圆周率
    Pi = np.pi
    # 存储映射结果
    dst = np.zeros((R, R, 3))
    src_h, src_w, _ = src.shape
    # 圆心
    x0, y0 = src_w//2, src_h//2

    for dst_y in range(0, R):

        theta =  Pi - (Pi/R)*dst_y
        temp_theta = math.tan(theta)**2

        for dst_x in range(0, R):
            # 取坐标点 p[i][j]
            # 计算 sita 和 fi

            phi = Pi - (Pi/R)*dst_x
            temp_phi = math.tan(phi)**2

            tempu = r/(temp_phi+ 1 + temp_phi/temp_theta)**0.5
            tempv = r/(temp_theta + 1 + temp_theta/temp_phi)**0.5

            if (phi < Pi/2):
                u = x0 + tempu
            else:
                u = x0 - tempu

            if (theta < Pi/2):
                v = y0 + tempv
            else:
                v = y0 - tempv

            if (u>=0 and v>=0 and u+0.5<src_w and v+0.5<src_h):
                dst[dst_y, dst_x, :] = src[int(v+0.5)][int(u+0.5)]

                # 计算在源图上四个近邻点的位置
                # src_x, src_y = u, v
                # src_x_0 = int(src_x)
                # src_y_0 = int(src_y)
                # src_x_1 = min(src_x_0 + 1, src_w - 1)
                # src_y_1 = min(src_y_0 + 1, src_h - 1)
                #
                # value0 = (src_x_1 - src_x) * src[src_y_0, src_x_0, :] + (src_x - src_x_0) * src[src_y_0, src_x_1, :]
                # value1 = (src_x_1 - src_x) * src[src_y_1, src_x_0, :] + (src_x - src_x_0) * src[src_y_1, src_x_1, :]
                # dst[dst_y, dst_x, :] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')

    return dst
if __name__ == '__main__':
    sg.theme('DarkAmber')   # 设置当前主题
    # 界面布局，将会按照列表顺序从上往下依次排列，二级列表中，从左往右依此排列
    layout = [  [sg.Text('实时图片',font=('黑体',15))],
                [sg.Image(size=(500,500),key = "-IMG-")],
                [sg.Text('鸡只数量                         时间',font=('黑体',15))],
                [sg.ML(size=(70,2),reroute_cprint=True,font=('黑体',15))],
                [sg.Button('计数')] ]
    
    # 创造窗口
    window = sg.Window('计数系统', layout)
    i=1
    # 事件循环并获取输入值
    while True:
        event, values = window.read()
        if event == None:
                break
        if event in (None, '计数'):   # 如果用户关闭窗口或点击`Canc
                url = "rtsp://admin:xz888888.@172.26.154.45/Streaming/Channels/1"
                #get_img_from_camera_net( url )
                frame = cv2.imread('1/'+str(i)+'.jpg')
                cut_img,R = cut(frame)
                result_img = undistort(cut_img,R)
                cv2.imwrite('1/12/'+str(i)+'.jpg',result_img)
                height,width = result_img.shape[:2]
                file1 = cv2.resize(result_img,(width//2,height//2),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite('1/12/'+str(i)+'.PNG',file1)
                weather_img_widget = window["-IMG-"]
                filename = '1/12/'+str(i)+'.PNG'
                weather_img_widget.Update(filename = filename)
                count()
                i=i+1
        '''
        if event in (None, '截图'):   # 如果用户关闭窗口或点击`Cancel
                frame = cv2.imread('1/1.jpg')
                cut_img,R = cut(frame)
                result_img = undistort(cut_img,R)
                cv2.imwrite('1/12/1.jpg',result_img)
                count()
        '''
        #sg.Button('计数'),                
        #print('You entered ', values[0])
    
    window.close()

