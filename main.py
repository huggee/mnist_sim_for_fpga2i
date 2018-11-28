# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from sklearn.datasets import fetch_mldata
from function import *
from mlp_without_bias import * # バイアス項なし
# from mlp_with_bias import * #バイアス項あり

def main():
    print(datetime.datetime.now())

    # パラメータ設定
    image_width = 28 # 画像の横サイズ
    image_height = 28 # 画像の縦サイズ
    n_h = 300 # 中間層のニューロン数
    eta = 0.1 # 学習率
    iteration = 3000  # 学習回数
    rep = 100 # 学習表示のインターバル
    coeff = 0.1 # 重み初期値(一様分布 -1 to +1)に掛かる係数
    tr = 30000 # Num of train data
    ts = 10000 # Num of test  data
    # trとtsの合計値が70000以下となるように設定

    # ネットワーク構築
    layers = np.array([image_height*image_width, n_h, 10]) # ニューロン数設定
    act = np.array(['sigmoid','sigmoid']) # 活性化関数設定
    ml = MultiLayer(layers, coeff) # インスタンス化
    print("* BUILDING MLP *")
    print("# of NEURONS: {} - {} - 10".format(image_height*image_width, n_h))
    print("ACTIVATION: (in-hidden){} (hidden-out){}".format(act[0], act[1]))
    print("")

    print("* TRAINING PARAM *")
    print("LEARNING RATE: {}".format(eta))
    print("MAX ITERATION: {}".format(iteration))
    print("")

    print("* IMAGE PARAM *")
    print("ORIGINAL SIZE (W x H): 28 x 28")
    print("AFTER RESIZING (W x H): {} x {}".format(image_width, image_height))
    print("# of IMAGE for TRAIN: {}".format(tr))
    print("# of IMAGE for TEST: {}".format(ts))
    print("")

    # MNISTデータセットをダウンロード
    print("* DOWNLOAD MNIST DATASET (if not exists) *")
    print("")
    mnist = fetch_mldata('MNIST original', data_home=".")

    print("* PRE-PROCESS MNIST DATASET *")
    print("")
    # train data (MNIST)
    p = np.random.randint(0, tr, tr) # 学習用画像のインデックスを生成
    X_origin = np.array( bit(mnist.data[p]/255.0) ) # 画像正規化
    X_3d = np.reshape(X_origin, [tr, 28, 28]) # 28x28の2D形式へ変換
    X_3d = [Image.fromarray(i) for i in X_3d] # 画像データへ変換
    X_resize = [i.resize((image_width, image_height)) for i in X_3d] # リサイズ
    X_resize = [np.asarray(i) for i in X_resize] # numpy形式へ変換
    X_resize = np.reshape(X_resize, [tr, image_width*image_height]) # 2Dから1Dへ変換
    y_origin = np.array( mnist.target[p] ) # 整数の教師データ(0-9)
    y_list = y_origin.tolist() # numpy配列をlist化
    y_int = [int(i) for i in y_list] # 各要素をintに変換
    y = np.eye(10)[y_int] # one-hot表現に変換

    X_train_origin = X_origin
    X_train = X_resize
    y_train = y

    # test data (MNIST)
    p = np.random.randint(tr, tr+ts, ts) # テスト用画像のインデックスを生成
    X_origin = np.array( bit(mnist.data[p]/255.0) ) # 画像正規化
    X_3d = np.reshape(X_origin, [ts, 28, 28]) # 28x28の2D形式へ変換
    X_3d = [Image.fromarray(i) for i in X_3d] # 画像データへ変換
    X_resize = [i.resize((image_width, image_height)) for i in X_3d] # リサイズ
    X_resize = [np.asarray(i) for i in X_resize] # numpy形式へ変換
    X_resize = np.reshape(X_resize, [ts, image_width*image_height]) # 2Dから1Dへ変換
    y_origin = np.array( mnist.target[p] ) # 整数の教師データ(0-9)
    y_list = y_origin.tolist() # numpy配列をlist化
    y_int = [int(i) for i in y_list] # 各要素をintに変換
    y = np.eye(10)[y_int] # one-hot表現に変換

    X_test_origin = X_origin
    X_test = X_resize
    y_test = y

    print("* VISUALIZE MNIST IMAGE *")
    # 学習用画像の最初の10枚を可視化
    plt.figure(figsize = (20, 4)) # 描画ウィンドウのサイズ
    for i in range(10):
        # オリジナルの28x28画像を1行目に描画
        ax = plt.subplot(2, 10, i+1) # 描画位置設定
        plt.title('28x28') # 各画像のタイトル
        plt.imshow(X_train_origin[i].reshape(28, 28)) # 2D画像に変換して描画
        plt.gray() # グレースケールで描画
        ax.get_xaxis().set_visible(False) # X軸不可視化
        ax.get_yaxis().set_visible(False) # Y軸不可視化
        xlim = ax.get_xlim() # X軸の範囲を取得
        ylim = ax.get_ylim() # Y軸の範囲を取得
    
        # リサイズ後の画像を2行目に描画
        ax = plt.subplot(2, 10, i+11) # 描画位置設定
        plt.title('%dx%d' % (image_width, image_height)) # 各画像のタイトル
        plt.imshow(X_train[i].reshape(image_height, image_width)) # 2D画像に変換して描画
        plt.gray() # グレースケールで描画
        ax.get_xaxis().set_visible(False) # X軸不可視化
        ax.get_yaxis().set_visible(False) # Y軸不可視化
        ### 描画されたリサイズ画像のセンタリング処理
        x_left = xlim[0] - (28.0 - image_width) / 2.0
        x_right = x_left + 28.0
        y_top = ylim[1] - (28.0 - image_height) / 2.0
        y_bottom = y_top + 28.0
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

    print("save mnist.png")
    print("")
    plt.savefig('mnist.png') # 画像をpng形式で保存
    # plt.show() # ウィンドウでも表示

    # train
    # plt.show()で画像ウィンドウを表示させた場合は、
    # 画像ウィンドウを閉じると学習が開始される
    print("* START TRAINING *")
    ml.fit(X_train, y_train, act, eta, iteration, rep)
    
    # test
    result_x = np.array(X_test) # listからnumpy配列に変換
    result_y = np.array(y_test) # listからnumpy配列に変換

    correct_n = 0
    print("* START TEST *")
    print(result_x.shape[0])
    for l in range(0,result_x.shape[0]):
        if all( result_y[l] == ml.result(result_x[l],act) ):
            correct_n += 1

    per = float( 100.0 * correct_n / ts )
    print("ACCURACY: {} %".format(per))


if __name__ == "__main__":
    main()
