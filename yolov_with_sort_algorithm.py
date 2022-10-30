#@title Set-up model. { run: "auto" }
checkpoint:str ="yolov6s" #@param ["yolov6s", "yolov6n", "yolov6t"]
device:str = "gpu"#@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}


import os, requests, torch, math, cv2
import numpy as np
import PIL
#Change directory so that imports wortk correctly
if os.getcwd()=="/content":
  os.chdir("YOLOv6")
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from matplotlib.pyplot import imshow
from scipy.optimize import linear_sum_assignment
from typing import List, Optional
from filterpy.kalman import KalmanFilter
#Download weights
if not os.path.exists(f"{checkpoint}.pt"):
  print("Downloading checkpoint...")
  os.system(f"""wget -c https://github.com/meituan/YOLOv6/releases/download/0.2.0/{checkpoint}.pt""")

#Set-up hardware options
cuda = device != 'cpu' and torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
  
def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def precess_image(path, img_size, stride, half):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    img_src = np.asarray(Image.open(path))
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    LOGGER.warning(e)
  from PIL import Image
  img_src = np.asarray(Image.open('G:/ECHO/YOLO6/data/images/image2.jpg'))
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src


model = DetectBackend(f"./{checkpoint}.pt", device=device)
stride = model.stride
class_names = load_yaml("./data/coco.yaml")['names']

if half & (device.type != 'cpu'):
  model.model.half()
else:
  model.model.float()
  half = False
img_size = (384,640)
if device.type != 'cpu':
  model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))
#@title Run YOLOv6 on an image from a URL. { run: "auto" }
url:str = "https://i.imgur.com/1IWZX69.jpg" #@param {type:"string"}
hide_labels: bool = False #@param {type:"boolean"}
hide_conf: bool = False #@param {type:"boolean"}



img_size:int = 640#@param {type:"integer"}

conf_thres: float =.5#@param {type:"number"}
iou_thres: float =.45 #@param {type:"number"}
max_det:int =  1000#@param {type:"integer"}
agnostic_nms: bool = False #@param {type:"boolean"}


img_size = check_img_size(img_size, s=stride)

def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

def colinearity(det,hist):
    '''
    det - current detection
    hist - last 2 mean detections
    '''
    dims = det[2:4] - det[:2]
    diag = np.sqrt(sum(dims**2))
    a = det[:2] + dims/2 - hist[-2]
    b = hist[-1] - hist[-2]
    len1 = np.sqrt(sum(a*a))
    len2 = np.sqrt(sum(b*b))
    ratio = len2/float(len1)
    maxdist = diag*(min(dims)/max(dims)+1)
    maxval = b.dot(b)
    a *= ratio
    return a.dot(b)/float(maxval) if maxval and maxdist > len1 else 0

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        #self.kf = SquareRootKalmanFilter(dim_x=7, dim_z=4)    ##see other filter https://filterpy.readthedocs.io/en/latest/
        #self.kf = ExtendedKalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cthist = [self.kf.x[:2].ravel()]

    def update(self, bbox, n):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.cthist.append(bbox[:2] + (bbox[2:4] - bbox[:2]) / 2)
        self.cthist = self.cthist[-n:]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
            self.kf.P *= 1.2 # we may be lost, increase uncertainty and responsiveness
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)



def associate_detections_to_trackers(detections, trackers, cost_fn = iou, threshold = 0.3):    ##default was 0.33
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    lendet = len(detections)
    lentrk = len(trackers)

    if(lentrk==0):
        return np.empty((0,2),dtype=int), np.arange(lendet), np.array([],dtype=int)
    cost_matrix = np.zeros((lendet,lentrk),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            cost_matrix[d,t] = cost_fn(det,trk)
    cost_matrix[cost_matrix < threshold] = 0.
    matched_indices = linear_sum_assignment(-cost_matrix)    #### here hungarian algorithm is used
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)

    costs = cost_matrix[tuple(matched_indices.T)] # select values from cost matrix by matched indices
    matches = matched_indices[np.where(costs)[0]] # remove zero values from matches
    unmatched_detections = np.where(np.in1d(range(lendet), matches[:,0], invert=True))[0]
    unmatched_trackers = np.where(np.in1d(range(lentrk), matches[:,1], invert=True))[0]

    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):
    def __init__(self,max_age=10 ,min_hits=0):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.tracked_obj = []
    def update(self, dets, cnum = 3): 
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          cnum - number of center positions to average
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        """
        self.frame_count += 1
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        ctmean = []
        to_del = []
        ret = []
        
        for t,trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            # print(pos)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   ##delete rows that contains any of NaN element
        for t in reversed(to_del):
            self.trackers.pop(t)    ##delete t th element (ID number to delete) from the list
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        
        ###for Unmatched Tracker
        for t in unmatched_trks:
            cnt = np.array(self.trackers[t].cthist)
            cnt = np.array([np.convolve(cnt[:,i], np.ones((cnum,))/cnum, mode='valid') for i in (0,1)]).T
            if cnt.shape[0] == 1: # fix same len
                cnt = np.concatenate((cnt,cnt),axis=0)
            ctmean.append(cnt)

        rematch, new_dets, lost_trks = associate_detections_to_trackers(dets[unmatched_dets],ctmean,colinearity,0.6)
        rematch = np.array([unmatched_dets[rematch[:,0]], unmatched_trks[rematch[:,1]]]).T
        matched = np.concatenate((matched, rematch.reshape(-1,2)))
        unmatched_dets = unmatched_dets[new_dets]
        unmatched_trks = unmatched_trks[lost_trks]

        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0], cnum+1)

        ##for Unmatched Detections
        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:]) 
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if((trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1],[trk.time_since_update])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)    ## deregistered
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))




(H, W) = (None, None)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (np.ceil((cumsum[N:] - cumsum[:-N]) / float(N))).astype("int")

##it is implemented in such way that takes least time 
##see timing details here : https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def numpy_ewma_vectorized_v2(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out.astype("int")





color = [[0,0,255],[255,0,0],[70,180,120],[255,255,0],[0,255,255],[255,0,255],[127,0,127],[127,127,0],[0,127,127],[255,127,127]]




mot_tracker = Sort(max_age =30, min_hits= 0)


# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# video=cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'),20, (1080,1920))
vs = cv2.VideoCapture('video.mp4')
fps = None
pause = 0 
trajectories = {}
# initBB = None
while True:
  _,frame = vs.read()
  # print(frame)
  if frame is None:
    break
  img_src = np.asarray(frame)
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  #image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.unsqueeze(dim = 0)
  image = image.permute(0, 3, 1,2)
  # print(image.shape)
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255
  img = image.to(device)

  if len(img.shape) == 3:
      img = img[None]
      # expand for batch dim
  pred_results = model(img)
  classes:Optional[List[int]] = 0 # the classes to keep
  det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

  gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
  img_ori = img_src.copy()
  box = [[.001,.002,.003,.004,.0001]]
  if len(det):
    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
  
    for *xyxy, conf, cls in reversed(det):
        # det = 
        class_num = int(cls)
        ##for filtering only ferson class
        if class_num == 0 :
          
          c = torch.Tensor([conf]).unsqueeze(dim = 0)
          box = torch.tensor(xyxy).view(1, 4).cpu()
          # box = torch.stack((box,cls), dim = 0)
          box = torch.cat((box,c.cpu()),dim  = 1).numpy()
          trackers = mot_tracker.update(box)
          mot_tracker.tracked_obj.append(trackers)
          # boxes.append(box)
          # print(box)
          for d in trackers:
            d = d.astype(np.int32)
            centroid = [int((d[0] + d[2]) / 2), int((d[1] + d[3]) / 2)]
            ID = d[4]
            
            text = "ID {}".format(ID)
            
            try:
                track = trajectories[str(ID)]
                
                track.append(centroid)
                
                ##applying moving average on centroid
                centx = [item[0] for item in track]
                centy = [item[1] for item in track]
                #centxx = running_mean(centx, 2)
                #centyy = running_mean(centy, 2)

                centxx = numpy_ewma_vectorized_v2(np.array(centx), 4)   #window size is 4
                centyy = numpy_ewma_vectorized_v2(np.array(centy), 4)

                mvavg = []
                for (i,j) in zip(centxx,centyy):
                    mvavg.append([i,j])
                trajectories[str(ID)] = track
                cv2.polylines(frame, [np.array(mvavg)], False, tuple(color[ID%10 - 1]), 4)
            except:
                trajectories[str(ID)] = [centroid]  
                track = trajectories[str(ID)]
                track.append(centroid)
                centx = [item[0] for item in track]
                centy = [item[1] for item in track]
                centxx = numpy_ewma_vectorized_v2(np.array(centx), 4)   #window size is 4
                centyy = numpy_ewma_vectorized_v2(np.array(centy), 4)

                mvavg = []
                for (i,j) in zip(centxx,centyy):
                    mvavg.append([i,j])
                trajectories[str(ID)] = track
            cv2.polylines(frame, [np.array(mvavg)], False, tuple(color[ID%10 - 1]), 4)
            # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
        label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
        Inferer.plot_box_and_label(frame, max(round(sum(frame.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
      
  # PIL.Image.fromarray(img_ori)
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) #& 0xFF
  # if key == ord("s") or key == ord('S'):
		# # select the bounding box of the object we want to track (make
  #       initBB = cv2.selectROI("Frame", frame, fromCenter=False,
  #              showCrosshair=True)
  # else :
  #   trackers = mot_tracker.update(np.array(box))
  # if initBB is not None:
  # 	# print(box)
  # 	# print(initBB)
  #   iou_ = iou((initBB[0],initBB[1],initBB[0]+initBB[2],initBB[1]+initBB[3]),box[0])
  #   if iou_ != 0 :
  #     video.write(frame)
      
  #     print('person_detected')
  #     del(frame)
cv2.destroyAllWindows()
# video.release()
print("Done")