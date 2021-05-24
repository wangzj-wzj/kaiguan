import torch
import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


def predict(img_path):
    net = torch.load('model.pkl') 
    net=net.cuda()
    torch.no_grad()
    img = PIL.Image.open(img_path)
    img_ = transform(img).unsqueeze(0)
    img_ = img_.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs,1)
    print(predicted)

def load_image(data_dir):
    datas = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        img = skimage.io.imread(fpath)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        img = img.astype(np.float32)/255.0
        datas.append(img)
        print('reading the images:%s' % fpath)
    datas = np.array(datas)
    return datas



def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


retinanet = torch.load('./model_final.pt')
if torch.cuda.is_available():
    retinanet=retinanet.cuda()

img_num = 0
for idx, data in enumerate(dataloader_val):
    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = retinanet(data['img'].float())
        print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores.cpu()>0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

        img[img<0] = 0
        img[img>255] = 255

        img = np.transpose(img, (1, 2, 0))

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[idxs[0][j]])]
            draw_caption(img, (x1, y1, x2, y2), label_name)
            print(x1,y1,x2,y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name)
            cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]

#       cv2.imshow('img', img)
#       cv2.waitKey(0)
        img_num += 1
        cv2.imwrite('./result_llh/photo_{}.jpg'.format(img_num),img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite('./result_llh/crop_{}.jpg'.format(img_num),cropped, [int( cv2.IMWRITE_JPEG_QUALITY), 95])


