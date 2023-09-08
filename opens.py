import cv2
import time 
import serial
import numpy as np

maps = [[1,0,2,0,3,0,4,0,5,0],
        [0,0,0,0,0,0,0,0,0,0],
        [6,0,7,0,8,0,9,0,10,0]]

maps = np.array(maps)
mf = (-1 < maps)


print("a")
serialP = serial.Serial('/dev/ttyS0', 115200, timeout=1)
print("a")
import math

def R_t(ass):

    pass
def L_t(ass):
    pass
def run(ass):
    pass
def Rrun(ass):
    pass
def Lrun(ass):
    pass
def stop(ass):
    pass

def expectedMovement(presentPosition:list, nextPosition:list):
    if presentPosition[0] < nextPosition[0]: #goto Top
        return 0
    if presentPosition[0] > nextPosition[0]: #goto Bottom
        return 180
    if presentPosition[0] == nextPosition[0]:
        if presentPosition[1] < nextPosition[1]: #Right
            return 270
        if presentPosition[1] < nextPosition[1]: #Left
            return 90
        
    print(None)



# import matplotlib.pyplot as plt
# import numpy as np

def solve_maze(maze, start, end):
    def is_valid(x, y):
        return 0 <= x < len(maze) and 0 <= y < len(maze[0])

    def is_traversable(x, y):
        return is_valid(x, y) and maze[x][y]

    stack = [(start[0], start[1])]
    visited = set()

    while stack:
        x, y = stack.pop()
        visited.add((x, y))

        if (x, y) == end:
            return maze  # 목표 위치에 도달하면 미로 상태 반환

        maze[x][y] = 2  # 지나간 길을 2로 표시

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_traversable(nx, ny) and (nx, ny) not in visited:
                stack.append((nx, ny))

    return None  # 목표 위치에 도달하지 못하면 None 반환



def drawsMaker(img, cornerPoint, ids):
    cornerPoint = np.array(cornerPoint, dtype=np.int32)

    for ic, v in enumerate(cornerPoint):
        i = v[0]
        img = cv2.line(img, i[0],i[1], (0,0,125))
        img = cv2.line(img, i[1],i[2], (0,0,125))
        img = cv2.line(img, i[2],i[3], (0,0,125))
        img = cv2.line(img, i[3],i[0], (0,0,125))
        img = cv2.line(img, i[3],i[1], (125,0,55),3)
        img = cv2.putText(img, f"{ids[ic]}", i[2]+[0,20], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232,123,0), 2)
    return img


# img = cv2.imread("board.png")
# corners, ids, markerDict = detector.detectMarkers(img)
# # print(corners)
# # img = drawsMaker(img, corners, ids)
# cv2.aruco.drawDetectedMarkers(img, corners, ids)

# print(ids.ravel())
# # print(markerDict[0].distance)
# cv2.imshow("ff", img)
# cv2.waitKey()

def printHello():
    print("hello")
class dataf:
    def __init__(self, fn, rimit) -> None:
        self.fn = fn
        self.distance = rimit
class ArucoMaker:
    def __init__(self, id, newcorners, distance):
        self.id = id
        self.corners = newcorners
        self.distance = distance
        self.centerX = int((self.corners[0][0] + self.corners[2][0]) * 0.5)
        self.centerY = int((self.corners[0][1] + self.corners[2][1]) * 0.5)
        rad = math.atan2((self.centerY - newcorners[0][1]), (self.centerX - newcorners[0][0]))
        deg = (rad*180)/math.pi
        deg -= 45
        if deg < -180:
            deg += 360
        self.deg = deg
def arucos(detector, img):
    
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is not None:
        imgs = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    else :
        imgs = img
    # cv2.imshow("ff", imgs)
    # cv2.waitKey(12)

    if ids is None:
        return imgs, None

    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    marker_size = 0.0275
    marker_objpts = np.array([[-marker_size/2,  marker_size/2, 0],
                [ marker_size/2,  marker_size/2, 0],
                [ marker_size/2, -marker_size/2, 0],
                [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
    camera_matrix = np.eye(3, 3, dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # cv2.solvePnP() 함수를 사용하여 카메라 매개변수를 계산합니다.
    rvec, tvec, _ = cv2.solvePnP(marker_objpts, corners[0], camera_matrix, dist_coeffs)
    # print(tvec)
    makers = {}
    for ic, v in enumerate(ids.ravel()):
        # 계산된 카메라 매개변수를 사용하여 거리를 계산합니다.
        x0 = corners[ic][0][0][1]
        y0 = corners[ic][0][0][0]
        z0 = tvec[0]
        x1 = corners[ic][0][1][1]
        y1 = corners[ic][0][1][0]
        z1 = tvec[1]

        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        makers[v] = ArucoMaker(v, corners[ic][0], distance)
        # print(v, ":" , distance)

    return imgs, makers

video = cv2.VideoCapture(0)
def command(detector, frame, config:dict):
    """
    최단 거리에 있는 마커의 커맨드를 동작 시킵니다. 
    커맨드는 config dict에 
    { id : dataf(fn, limit) ,  }
    형식으로 입력해주세요.
    """
    global theLook, timeLook
    if not ("tick" in config) :
        raise ValueError
    
    
    maker_img, makers = arucos(detector, frame)
    action = None
    lens = 10000
    if makers is not None:
        for key in config:
            if key in makers:
                if makers[key].distance > config[key].distance:
                    if makers[key].distance < lens:
                        action = key            
                        lens = makers[key].distance
                        
        
    if action is not None:
        if time.time() - timeLook > 3 and theLook:
            config[action].fn()
            timeLook = time.time()
        theLook = False
        return True
    else:
        theLook = True
        return False


config = {
    0 : dataf(printHello, 100),


    "tick" : None
}

# model = tf.keras.models.load_model("model.keras")

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
theLook = True
timeLook = time.time()

label = {
    0: 0, 1:45, 2:90
}

def centerConneter(fx, fy, fdeg, constX, constY, constDeg):
    errX = constX - fx
    errY = constY - fy

    errD = constDeg - fdeg
    ex = 0
    ey = 0
    erd = 0.0
    if errX < -30:
        ex = 0
    elif errX > 30:
        ex = 2
    else: ex = 1

    if errY < -30:
        ey = 0
    elif errY > 30:
        ey = 2
    else: ey = 1

    if fdeg < constDeg+10   :
        erd = 1
    elif fdeg > constDeg+10:
        erd = 0
    else:
        erd = 2

    if ex == 1:
        serialP.write('d|150'.encode())
    elif ex == 2:
        serialP.write('a|150'.encode())
    else: serialP.write('p|0'.encode())

    if erd == 0:
        serialP.write('l|150'.encode())
    elif erd == 1:
        serialP.write('r|150'.encode())
    elif erd == 2:
        serialP.write('p|0'.encode())

    if erd ==2 and ex == 2:
        return True
    return False

def runs(x, rx):
    if rx - x < -10:
        serialP.write('d|128'.encode())
        time.sleep(0.12)
        serialP.write('p|0'.encode())

    if rx - x > 10:
        serialP.write('a|128'.encode())
        time.sleep(0.12)
        serialP.write('p|0'.encode())

    


def trun(dex, deg):
    if dex > deg+10 :
        print("sd")
        serialP.write('r|128'.encode())
        time.sleep(0.12)
        serialP.write('p|0'.encode())
        return False
    elif dex < deg-10:
        serialP.write('l|128'.encode())
        time.sleep(0.12)
        serialP.write('p|0'.encode())
        return False
    return True
a = input("시작 ID ")
print(a)
print(maps == (int(a)))
a = np.where(maps == (int(a)))
print(a)
a = (a[0][0], a[1][0])


b = input("끝 ID ")
b = np.where(maps == int(b))
b = (b[0][0], b[1][0])
print(b)
b = 0
result = solve_maze(np.array(mf, np.uint8), (0,0),(2,9))
print(result)
if result is not None:    
    pap = np.where(np.array(result, np.uint8)==2)
    # print(np.where(np.array(result, np.uint8)==2))
    def nextSeed():
        global b 

        pos1 = pap[0][b*2] 
        pos2 = pap[1][b*2] 
        
        pos3 = pap[0][(b+1)*2] 
        pos4 = pap[1][(b+1)*2] 
        b += 1
        return (pos1, pos2), (pos3, pos4) 

    
    a, c = nextSeed()
    tg = -10
    cyc = c

    while True:
        _, f = video.read()
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        print(maps[a[0],a[1]])
        print(expectedMovement(cyc, (a[0],a[1])), "asd")
        
        # T0F = command(detector, f, config)
        # if not T0F:
        #     pred = model.predict([f])[0]
        #     deg = label[np.argmax(pred)]
        #     siri_Ya.write(f"{deg}|{110}".encode())

        ___, asas = arucos(detector, f)
        id = maps[a[0],a[1]]
        cyc = (0,0)
        print((a[0],a[1]), cyc)
        cv2.imwrite("aaa.png", ___)
        
        if asas is not None:
            if id in asas:

                print(asas[id].deg+180)
                print(f"{asas[id].centerX=}", f"{asas[id].centerY=}")
                aT = trun(int(asas[id].deg+180), expectedMovement((a[0],a[1]), cyc))
                aR = runs(asas[id].centerX,320 )
                if aT and aR:
                    a = nextSeed()
                    cyc = (a[0],a[1])
                time.sleep(0.2)

        else:
            print("aa")
            serialP.write('w|130'.encode())
