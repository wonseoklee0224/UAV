import numpy as np
import random

a = 9.61
b = 0.16
f_c = 2000000000
Los = 3
NLos = 23
B = 1000000
N_0 = -170
P_max = 20
RTT = 0.03
FS = 8000000
MSS = 11680
c_1 = 1.120
c_2 = 4.6746


def db_to_w(tmp):
    return 10 ** (tmp / 10)


def w_to_db(tmp):
    return 10 * np.log10(tmp)


def distance_cal(UAV_H, UAV_Y, UAV_X, user_point):
    d = np.sqrt((UAV_H - 0) ** 2 + (UAV_Y - user_point[1]) ** 2 + (UAV_X - user_point[2]) ** 2)
    return d


def degree_cal(distance, UAV_H):
    degree_value = np.arcsin(UAV_H / distance) * 180 / np.pi
    return degree_value


def P_Los_cal(degree):
    P_loss = 1 / (1 + a * np.exp(-b * (degree - a)))
    return P_loss


def gain_cal(distance, P_loss):
    gain = -20 * np.log10(distance) - 20 * np.log10(f_c) + 147.55 - w_to_db(
        db_to_w(P_loss * Los) + db_to_w((1 - P_loss) * NLos))
    return gain


def rate_cal(B_k, p_t, gain):
    rate = B_k * np.log2(1 + db_to_w((p_t + gain - 30) - (w_to_db(B_k) + N_0 - 30)))
    return rate


def L_cal(rate):
    L_1 = np.log2(rate * RTT / MSS + 1) - 1
    L_2 = np.log2(FS / 2 * MSS + 1) - 1
    L = np.minimum(L_1, L_2)
    return L


def MOS_cal(rate, L):
    tmp = (3 * RTT) + (FS / rate) + L * (MSS / rate) + RTT - 2 * MSS * (2 ** L - 1) / rate
    MOS = -1 * c_1 * np.log(tmp) + c_2
    return MOS


class Environment:
    def __init__(self, GRID_size, user_point):
        self.GRID_H = GRID_size[0]
        self.GRID_Y = GRID_size[1]
        self.GRID_X = GRID_size[2]
        self.USER_points = user_point
        # self.nbStates = self.gridSize * self.gridSize  # 필요할지 확인 필요
        self.state = np.empty(3, dtype=np.uint8)  # uav_point, user_point 2개로 변경해야 할듯하다.
        self.old_MOS = 0

    def reset(self):
        initialUAV_H = random.randrange(100, self.GRID_H + 1)
        initialUAV_Y = random.randrange(1, self.GRID_Y + 1)
        initialUAV_X = random.randrange(1, self.GRID_X + 1)
        self.state = np.array([initialUAV_H, initialUAV_Y, initialUAV_X])
        print("초기", self.state)
        return self.get_state()

    def get_state(self):
        stateInfo = self.state
        UAV_H = stateInfo[0]
        UAV_Y = stateInfo[1]
        UAV_X = stateInfo[2]
        UAV_position = [UAV_H, UAV_Y, UAV_X]
        return UAV_position

    def updateState(self, action):
        UAV_H, UAV_Y, UAV_X = self.get_state()

        if action == 1:
            if UAV_H < self.GRID_H:
                UAV_H += 1
        elif action == 2:
            if UAV_H > 0:
                UAV_H -= 1
        elif action == 3:
            if UAV_Y < self.GRID_Y:
                UAV_Y += 1
        elif action == 4:
            if UAV_Y > 0:
                UAV_Y -= 1
        elif action == 5:
            if UAV_X < self.GRID_X:
                UAV_X += 1
        elif action == 6:
            if UAV_X > 0:
                UAV_X -= 1

        self.state = UAV_H, UAV_Y, UAV_X


    def getReward(self):
        UAV_H, UAV_Y, UAV_X = self.get_state()
        User_points = self.USER_points
        n = len(self.USER_points)
        B_k = B  # UAV number로 나눠저야한다.
        p_t = w_to_db(db_to_w(P_max) / n)
        result = 0
        for x in range(0, n):
            distance = distance_cal(UAV_H, UAV_Y, UAV_X, User_points[x])
            degree = degree_cal(distance, UAV_H)
            P_loss = P_Los_cal(degree)
            gain = gain_cal(distance, P_loss)
            rate = rate_cal(B_k, p_t, gain)
            L = L_cal(rate)
            MOS = MOS_cal(rate, L)
            result += MOS
            x += 1
        if self.old_MOS < result:
            self.old_MOS = result
            return 10
        elif self.old_MOS == result:
            self.old_MOS = result
            return 1
        elif self.old_MOS > result:
            self.old_MOS = result
            return -10

    def act(self, action):
        self.updateState(action)
        reward = self.getReward()
        MOS_value = self.old_MOS
        return reward, self.get_state(), MOS_value
