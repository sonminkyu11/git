import numpy as np
import matplotlib.pyplot as plt
import math

class LaneChange():
  def __init__(self):
    self.lc = LaneChange()
    self.v = 0            
    self.a_max = 2        
    self.obs_left_data = np.zeros((2,3)) # obs_left_data[0:1] = [rear, front], obs_left_data[][1:3] = [vel, distance, acc]  
    self.obs_pres_data = np.zeros((2,3))
    self.obs_right_data = np.zeros((2,3))
    self.wide = 2
    self.width = 2
    self.real_path = 1
    self.s_cnt = 0            # distance to travel count
    self.s = 0

  
  def path_generator(self):
    """
    input 
    qi = difference in width from global path => present path
    qf[3] = difference in width from global path => last able paths[0:2] = left, pres, right

    return
    qlist[3][2] = actual able paths coordinates => qlist[][0:1] = y, x

    """
    
    s_min = 1                #heuristic constant
    s_max = 50               #heuristic constant
    a_max = 2                #sub // self.ego_ay

    if s_min + self.v**2/a_max < s_max:
      sf = s_min + self.v**2/a_max   # sf = distance to the end of lane change
    else:
      sf = s_max

    qi = 1                  
    qf = list()             
    qf = [qi - self.width, qi, qi + self.width]
    qlist = list()          
    N = 0.2                 
    s = np.arange(0, sf + N, N)

    for i in range(3):    # spline interporlation for lane change
      a = (2/(sf**3))*(qi - qf[i])
      b = 3*(qf[i] - qi)/(sf**2)

      q = (a*s**3)+(b*s**2) + qi

      # plt.figure(figsize = (16, 5))
      # plt.subplot(121)  
      # plt.plot(s, q, 'r')
      # plt.grid()
      data = np.array([s, q])
      
      qlist.append(data)
      
      return qlist

  def dynamic_cost(self):

    d_cost = np.zeros((3))        
    dynamic_weight = 20                # heuristic constant

    d_cost[0] = 1/(10+self.obs_left_data[1][0])
    d_cost[1] = 1/(10+self.obs_pres_data[1][0])
    d_cost[2] = 1/(10+self.obs_right_data[1][0])

    return d_cost*dynamic_weight

  def global_cost(self):

    gbl_cost = self.qf/np.sum(self.qf)
    gbl_weight = 0            # heuristic constant
    dis = 1600                # sub, distance from intersection
    
    if dis > 1500:
      gbl_weight = 0.5
    elif dis > 150:
      gbl_weight = 1
    else:
      gbl_weight = 100

    return gbl_cost*gbl_weight

  def lc_cost(self):
    
    lane_change_cost = 0.02        #heuristic constant
    lc_weight = 1               #heuristic constant

    return lane_change_cost*lc_weight
  
  def cost_func(self):
    """
    input
    each cost function (dynamic, lc, global) => ovral_cost

    return
    want_lane[3] = best lane that is derived from cost function => want_lane[0:2] = 0 is best, 2 is worst
    """

    ovral_cost = list()
    ovral_cost.append((self.global_cost()[0] + self.lc_cost() + self.dynamic_cost()[0], 0)) # left lane
    ovral_cost.append((self.global_cost()[1] + self.dynamic_cost()[1], 1))                     # pres lane
    ovral_cost.append((self.global_cost()[2] + self.lc_cost() + self.dynamic_cost()[2], 2))   # right lane

    # print(ovral_cost)

    ovral_cost.sort(key = lambda x:x[0])

    # print(ovral_cost)

    want_lane = list()
    for i in range(3):
      want_lane.append(ovral_cost[i][1])

    # print(want_lane)

    return want_lane

  def avoid(self):
    """
    input
    cost_func()

    return
    self.real_path => that
    """


    safe_d = max(self.v*3.6-15, 5)                # safe distance ,  *3.6 km/s
    
    want_lane = self.cost_func()

    lc_time = 2                         # heuristic constant
    delta_dis = np.zeros((3, 2))        # distance after LC


    rel_left_data = [[self.v - self.obs_left_data[0][0], self.obs_left_data[0][1] + safe_d + self.wide], [self.v - self.obs_left_data[1][0], self.obs_left_data[1][1] - safe_d - self.wide]]    
    rel_pres_data = [[self.v - self.obs_pres_data[0][0], self.obs_pres_data[0][1] + safe_d + self.wide], [self.v - self.obs_pres_data[1][0], self.obs_pres_data[1][1] - safe_d - self.wide]]
    rel_right_data = [[self.v - self.obs_right_data[0][0], self.obs_right_data[0][1] + safe_d + self.wide], [self.v - self.obs_right_data[1][0], self.obs_right_data[1][1] - safe_d - self.wide]]

    delta_dis[0][0] = -1*((-self.obs_left_data[0][1]-self.obs_left_data[0][0]*lc_time)+math.sqrt(((self.acc*(lc_time**2)/2)+self.v*lc_time)**2-(self.width)**2))+safe_d+self.wide       # delta_dis[0:2][:] = left pres right, delta_dis[:][0:1] = 뒤 앞
    delta_dis[0][1] = (self.obs_left_data[1][1]+self.obs_left_data[1][0]*lc_time)-math.sqrt(((self.acc*(lc_time**2)/2)+self.v*lc_time)**2-(self.width)**2)-safe_d-self.wide

    delta_dis[1][0] = -1*((-self.obs_pres_data[0][1]-self.obs_pres_data[0][0]*lc_time)+((self.acc*(lc_time**2)/2)+self.v*lc_time))+safe_d+self.wide
    delta_dis[1][1] = (self.obs_pres_data[1][1]+self.obs_pres_data[1][0]*lc_time)-((self.acc*(lc_time**2)/2)+self.v*lc_time)-safe_d-self.wide

    delta_dis[2][0] = -1*((-self.obs_right_data[0][1]-self.obs_right_data[0][0]*lc_time)+math.sqrt(((self.acc*(lc_time**2)/2)+self.v*lc_time)**2-(self.width)**2))+safe_d+self.wide
    delta_dis[2][1] = (self.obs_right_data[1][1]+self.obs_right_data[1][0]*lc_time)-math.sqrt(((self.acc*(lc_time**2)/2)+self.v*lc_time)**2-(self.width)**2)-safe_d-self.wide

    # print(delta_dis)

    for i in range(3):
      if want_lane[i] == 0:
        if rel_left_data[0][1] < 0 and rel_left_data[0][1] > 0 and delta_dis[0][0] < 0 and delta_dis[0][1] > 0:
          self.real_path = 0

          break

        else:
          pass

      elif want_lane[i] == 2:
        if rel_right_data[0][1] < 0 and rel_right_data[0][1] > 0 and delta_dis[2][0] < 0 and delta_dis[2][1] > 0:
          self.real_path = 2

          break

        else:
          pass
      else:
        self.real_path = 1
        
        break

  def go_trigger(self):
    go_path = self.path_generator()[self.real_path]

    return go_path

  def timer_callback(self):
    
    try:

      if self.s % 10 > self.cnt :
        self.cnt += 1
        self.avoid()

        if self.real_path == 1:
          pass
        else:
          go_path = self.path_generator()[self.real_path]

      else:
        pass

    except Exception as e: 
      print('code error: ', e) 