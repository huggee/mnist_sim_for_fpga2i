'''
MLP without bias
'''

import math
import numpy as np
import sys
import random
from sklearn.datasets import fetch_mldata
import time
from function import *

class MultiLayer:
    def __init__(self, struct, coeff):
        self.layers = struct.size
        self.str = np.array(struct)

        #重み初期値の生成
        weight1 = np.zeros(0,dtype=np.float128)
        self.weight = []
        for var1 in range(0,self.layers-1):
            weight1 = bit(coeff * np.random.uniform(-1, 1, (struct[var1], struct[var1+1] )))
            self.weight.append(weight1)

        # 隠れ層の活性の生成
        self.hid = []
        for var1 in range(0,self.layers-2):
            hid1 = np.zeros(self.str[var1+1],dtype=np.float128)
            self.hid.append(hid1)

        # 修正情報の生成
        self.PHI = []
        for var1 in range(0,self.layers-1):
            phi1 = np.zeros( struct[var1+1] + 0 ,dtype=np.float128)
            self.PHI.append(phi1)



    def fit(self, X, T, act, eta, iteration,rep):
        #X = np.hstack([np.ones([X.shape[0], 1]), X])
        T = np.array(T)
        start_at     = time.time()
        cur_at       = start_at
        loss_acc     = 0
        for ite in range(0,iteration):
            # 入力データをランダムに1つ取り出す。
            i = np.random.randint(X.shape[0])
            x = np.array(X[i],dtype=np.float128)
            t = T[i]

            if (ite + 1) % rep == 0:
                now = time.time()
                print('{}/{}, Thr.= {:.2f} iter/sec, Loss = {:.8f}'.format(ite+1,iteration,rep/(now-cur_at),loss_acc/rep))
                cur_at = now
                loss_acc = 0


        #　順伝播計算
            for var1 in range(0,self.layers-1):
                # INPUT to HIDDEN
                if var1 == 0:
                    w = bit(np.array(self.weight[0]))
                    self.hid[0] = act_func( multiply_inf(x,w),act[var1] )

                # HIDDEN to OUTPUT
                elif var1 == self.layers-2:
                    w = bit(np.array(self.weight[var1]))
                    self.y = act_func( multiply_inf( self.hid[var1-1], w ),act[var1] )

                # HIDDEN to HIDDEN
                else:
                    w = bit(np.array(self.weight[var1]))
                    self.hid[var1] = act_func( multiply_inf( self.hid[var1-1], w ),act[var1] )


        # 逆伝播計算 + 重み更新
            for var1 in range(0,self.layers-1):
                inv_lay = self.layers-2-var1

                # OUTPUT to HIDDEN
                if var1 == 0:
                    ph = np.array(self.PHI[inv_lay])
                    w = np.array(self.weight[inv_lay])

                    ph_ = np.array(self.PHI[inv_lay-1])

                    # 修正情報
                    ph = bit_16(dif(self.y,act[inv_lay]) * ( t - self.y ))
                    loss_acc += 0.5 * (np.sum(np.dot(t-self.y,t-self.y)))
                    # 重み更新
                    for var2 in range(0,self.hid[inv_lay-1].size):
                        w[var2] += bit_16(eta * ph * self.hid[inv_lay-1][var2])

                    # PHI逆伝播
                    for val2 in range(0,self.hid[inv_lay-1].size):
                        #print(val2)
                        ph_[val2] = bit_16(dif(self.hid[inv_lay-1][val2],act[inv_lay-1]) * multiply_lea(ph , w[val2]))

                    # Over write
                    self.PHI[inv_lay] = np.array(ph)
                    self.weight[inv_lay] = np.array(w)
                    self.PHI[inv_lay - 1] = np.array(ph_)

                # HIDDEN to INPUT
                elif var1 == self.layers-2:
                    ph2 = np.array(self.PHI[inv_lay]) # PHI算出用

                    ph = np.array(self.PHI[inv_lay-1])
                    w = np.array(self.weight[inv_lay])


                    # 重み算出
                    for var2 in range(0,x.size):
                        w[var2] += bit_16(eta * ph2 * x[var2])

                    # Over write
                    self.PHI[inv_lay-1] = np.array(ph)
                    self.weight[inv_lay] = np.array(w)



                # HIDDEN to HIDDEN
                else:
                    ph2 = np.array(self.PHI[inv_lay]) # PHI算出用

                    ph = np.array(self.PHI[inv_lay-1])
                    w = np.array(self.weight[inv_lay])

                    # 重み算出
                    for var2 in range(0,self.hid[inv_lay-1].size):
                        w[var2] += bit_16(eta * ph2 * self.hid[inv_lay-1][var2])

                    # PHI算出
                    for val2 in range(0,self.hid[inv_lay-1].size):
                        ph[val2] = bit_16(dif(self.hid[inv_lay-1][val2],act[inv_lay-1]) * multiply_lea(ph2 , w[val2]))


                    # Over write
                    self.PHI[inv_lay-1] = np.array(ph)
                    self.weight[inv_lay] = np.array(w)


    def result(self, l, act):
        x = l
        for var1 in range(0,self.layers-1):
            # INPUT to HIDDEN
            if var1 == 0:
                w = bit(np.array(self.weight[0]))
                self.hid[0] = act_func( multiply_inf(x,w),act[var1] )

            # HIDDEN to OUTPUT
            elif var1 == self.layers-2:
                w = bit(np.array(self.weight[var1]))
                self.y = act_func( multiply_inf( self.hid[var1-1], w ),act[var1])

            # HIDDEN to HIDDEN
            else:
                w = bit(np.array(self.weight[var1]))
                self.hid[var1] = act_func( multiply_inf( self.hid[var1-1], w ),act[var1] )

        return step(self.y)
