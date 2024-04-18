  
# Script file to hide implementation details for PyTorch computer vision module

import builtins
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile 

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Python에서 MNIST 데이터셋을 불러와서 처리하는 과정
# 이 함수를 실행하면, builtins 모듈을 통해 전역 변수로 설정된 data_train, data_test, train_loader, test_loader가 생성되어 어디서든 접근할 수 있게 됩니다. 이러한 설정은 함수 내에서 데이터를 처리하고, 이후에 다른 부분에서 해당 데이터를 사용할 때 유용하게 활용될 수 있음

def load_mnist(batch_size=64): # load_mnist라는 이름의 함수를 정의하고, 이 함수는 기본적으로 batch_size 매개변수를 64로 설정합니다. 이 매개변수는 데이터를 얼마나 많은 단위로 나눌지 결정
    builtins.data_train = torchvision.datasets.MNIST('./data',
        download=True,train=True,transform=ToTensor()) # torchvision 라이브러리의 datasets 모듈을 사용하여 MNIST 데이터셋을 불러옵니다. './data'는 데이터셋이 저장될 경로를 지정하며, download=True는 해당 경로에 데이터가 없을 경우 인터넷에서 자동으로 다운로드하도록 설정합니다. train=True는 학습용 데이터셋을 불러오는 것을 의미하고, transform=ToTensor()는 데이터셋의 이미지들을 파이토치 텐서로 변환하는 함수를 적용
    builtins.data_test = torchvision.datasets.MNIST('./data', 
        download=True,train=False,transform=ToTensor()) # 테스트 데이터셋을 불러오는 코드입니다. train=False로 설정하여 학습용이 아닌 테스트용 데이터셋을 불러옴
    builtins.train_loader = torch.utils.data.DataLoader(data_train,batch_size=batch_size) # 학습 데이터셋을 데이터 로더에 로드합니다. 데이터 로더는 데이터셋을 지정된 배치 크기에 맞게 나누고, 이를 반복 가능한 객체로 만들어 학습 과정에서 쉽게 사용할 수 있게 도움
    builtins.test_loader = torch.utils.data.DataLoader(data_test,batch_size=batch_size)

# 신경망을 한 에폭(epoch) 동안 학습하는 과정을 구현한 Python 함수
# 이 함수는 모델을 학습시키고, 각 배치에서의 평균 손실과 정확도를 계산하여 반환하는데 이를 통해 학습 과정을 모니터링할 수 있음

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss()): # 이 함수는 여러 매개변수를 받는데, net은 학습할 신경망 모델, dataloader는 데이터 로더, lr은 학습률(기본값 0.01), optimizer는 최적화 도구(기본값은 None), loss_fn은 손실 함수로 기본적으로 Negative Log Likelihood Loss를 사용
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr) # 최적화 도구가 제공되지 않았다면, Adam 최적화 도구를 사용하여 신경망의 매개변수를 최적화하며, 학습률은 lr로 설정
    net.train() # 모델을 학습 모드로 설정합니다. 이는 일부 신경망 계층(예: 드롭아웃 계층)이 학습과 평가 모드에서 다르게 동작하기 때문에 필요
    total_loss,acc,count = 0,0,0 # 총 손실, 정확도, 처리한 샘플 수를 초기화
    for features,labels in dataloader: # 데이터 로더로부터 특징(feature)과 레이블(label)을 반복적으로 가져옴
        optimizer.zero_grad() # 최적화 도구의 모든 기울기를 0으로 초기화하는데 새로운 가중치 업데이트를 위해 필수
        lbls = labels.to(default_device) # 레이블을 기본 계산 장치(예: GPU)로 이동
        out = net(features.to(default_device)) # 특징을 같은 장치로 이동시킨 후, 신경망을 통해 예측을 수행
        loss = loss_fn(out,lbls) #cross_entropy(out,labels) 예측 결과와 레이블을 이용해 손실을 계산
        loss.backward() # 손실에 대한 기울기를 계산
        optimizer.step() # 계산된 기울기를 이용해 신경망의 가중치를 업데이트
        total_loss+=loss # 총 손실을 누적
        _,predicted = torch.max(out,1) # 예측된 결과 중 가장 높은 확률을 가진 클래스를 선택
        acc+=(predicted==lbls).sum() # 정확하게 예측된 수를 누적
        count+=len(labels) # 처리된 레이블의 수를 누적
    return total_loss.item()/count, acc.item()/count # 평균 손실과 정확도를 반환

# 주어진 신경망 모델을 평가하는 과정을 나타내는 Python 함수
# 이 함수는 주어진 데이터 로더를 사용하여 모델의 성능을 평가하고, 평균 손실과 정확도를 반환하여 모델의 효율성을 확인

def validate(net, dataloader,loss_fn=nn.NLLLoss()): # validate 함수를 정의하고, 세 개의 매개변수를 받는데 net은 평가할 신경망 모델, dataloader는 평가 데이터셋을 로딩하는 데이터 로더, loss_fn은 손실 함수로 기본값은 Negative Log Likelihood Loss
    net.eval() # 모델을 평가(evaluation) 모드로 설정하는데 이 모드에서는 모델의 학습 과정에만 적용되는 특정 기능들(예: 드롭아웃)이 비활성화
    count,acc,loss = 0,0,0 # 총 처리한 데이터의 수, 정확히 예측된 데이터의 수, 그리고 총 손실을 0으로 초기화
    with torch.no_grad(): # 기울기 계산이 수행되지 않는데 평가 시에는 모델의 가중치가 갱신되지 않음
        for features,labels in dataloader: # 데이터 로더에서 데이터 배치를 반복적으로 가져오는데 각 배치는 features (특징 데이터)와 labels (레이블 데이터)로 구성
            lbls = labels.to(default_device) # 레이블을 기본 계산 장치(예: GPU)로 이동
            out = net(features.to(default_device)) # 특징 데이터도 같은 계산 장치로 이동한 후, 신경망 모델을 통해 예측을 수행
            loss += loss_fn(out,lbls) # 예측 결과와 실제 레이블을 사용하여 손실을 계산하고, 총 손실에 누적
            pred = torch.max(out,1)[1] # 신경망의 출력에서 가장 높은 값을 가진 클래스의 인덱스를 추출하여 예측 결과로 사용
            acc += (pred==lbls).sum() # 예측이 정확했던 샘플의 수를 누적
            count += len(labels) # 처리한 레이블의 총 수를 누적
    return loss.item()/count, acc.item()/count # 평균 손실과 평균 정확도를 계산하여 반환

# 신경망 모델을 여러 에폭(epoch) 동안 학습하고 평가하는 과정을 정의하는 Python 함수

def train(net,train_loader,test_loader,optimizer=None,lr=0.01,epochs=10,loss_fn=nn.NLLLoss()): # train 함수를 정의 - net: 학습할 신경망 모델; train_loader와 test_loader: 학습 및 테스트 데이터셋을 로드하는 데 사용되는 데이터 로더; optimizer: 최적화 도구 (기본적으로 None이며, 지정되지 않았을 경우 Adam 최적화 도구가 사용); lr: 학습률 (기본값은 0.01); epochs: 전체 학습 과정을 반복할 횟수 (기본값은 10); loss_fn: 손실 함수 (기본값은 Negative Log Likelihood Loss)
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr) # 최적화 도구 (기본적으로 None이며, 지정되지 않았을 경우 Adam 최적화 도구가 사용하며, 학습률은 lr로 설정)
    res = { 'train_loss' : [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # 학습 및 검증 과정에서 계산된 손실과 정확도를 저장할 딕셔너리를 초기화
    for ep in range(epochs): # 지정된 에폭 수만큼 반복
        tl,ta = train_epoch(net,train_loader,optimizer=optimizer,lr=lr,loss_fn=loss_fn) # train_epoch 함수를 호출하여 한 에폭 동안의 학습을 수행하고, 학습 손실(tl)과 정확도(ta)를 반환 받음
        vl,va = validate(net,test_loader,loss_fn=loss_fn) # validate 함수를 호출하여 모델을 검증 데이터셋에 대해 평가하고, 검증 손실(vl)과 정확도(va)를 반환 받음
        print(f"Epoch {ep:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}") # 에폭, 학습 정확도, 검증 정확도, 학습 손실, 검증 손실을 출력
        res['train_loss'].append(tl) # 각 결과 값을 딕셔너리에 추가
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)
    return res # 학습과 검증 과정에서의 결과를 담은 딕셔너리를 반환

# 신경망 모델을 학습하면서 주기적으로 학습 상태를 출력하고, 각 에폭의 끝에서 검증 성능을 출력하는 Python 함수

def train_long(net,train_loader,test_loader,epochs=5,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss(),print_freq=10): # train_long 함수를 정의 - net: 학습할 신경망 모델; train_loader, test_loader: 각각 학습과 검증 데이터를 로드하는 데 사용되는 데이터 로더; epochs: 전체 학습을 반복할 횟수, 기본값은 5; lr: 학습률, 기본값은 0.01; optimizer: 최적화 도구, 기본적으로 None이며, 제공되지 않았을 경우 Adam 최적화 도구를 사용; loss_fn: 손실 함수, 기본값은 Negative Log Likelihood Loss); print_freq: 학습 상태를 출력할 빈도
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr) # 최적화 도구가 제공되지 않은 경우, Adam 최적화 도구를 사용
    for epoch in range(epochs): # 지정된 횟수만큼 에폭을 반복
        net.train() # 모델을 학습 모드로 설정
        total_loss,acc,count = 0,0,0 # 총 손실, 정확도 계산을 위한 누적된 정확도, 그리고 처리된 데이터 수를 초기화
        for i, (features,labels) in enumerate(train_loader): # 학습 데이터 로더로부터 데이터 배치를 가져옴
            lbls = labels.to(default_device) # 레이블을 기본 계산 장치로 이동
            optimizer.zero_grad() # 기울기 버퍼를 0으로 초기화
            out = net(features.to(default_device)) # 특징 데이터를 기본 계산 장치로 이동시킨 후, 모델을 통해 예측을 수행
            loss = loss_fn(out,lbls) # 예측 결과와 레이블을 사용하여 손실을 계산
            loss.backward() # 손실에 대한 기울기를 계산
            optimizer.step() # 계산된 기울기를 사용하여 모델의 가중치를 갱신
            total_loss+=loss # 총 손실을 누적
            _,predicted = torch.max(out,1) # 예측된 결과 중 가장 높은 값을 가진 클래스의 인덱스를 추출
            acc+=(predicted==lbls).sum() # 정확하게 예측된 샘플 수를 누적
            count+=len(labels) # 처리된 레이블 수를 누적
            if i%print_freq==0: # 지정된 빈도마다 현재까지의 학습 상태를 출력
                print("Epoch {}, minibatch {}: train acc = {}, train loss = {}".format(epoch,i,acc.item()/count,total_loss.item()/count))
        vl,va = validate(net,test_loader,loss_fn) # 에폭이 끝날 때마다 검증 함수를 호출하여 검증 손실과 정확도를 계산
        print("Epoch {} done, validation acc = {}, validation loss = {}".format(epoch,va,vl)) # 에폭의 학습 결과와 검증 결과를 출력

# 학습 및 검증 데이터에 대한 정확도와 손실을 시각화하는 Python 함수인데 Matplotlib 라이브러리를 사용하여 결과를 그래프로 표시

def plot_results(hist): # plot_results라는 함수를 정의하는데 hist라는 이름의 딕셔너리를 매개변수로 받는데 학습과 검증 과정의 정확도와 손실이 배열 형태로 저장되어 있음
    plt.figure(figsize=(15,5)) # 새로운 그래프 창을 만들고, 크기를 가로 15인치, 세로 5인치로 설정
    plt.subplot(121) # 두 개의 그래프를 나란히 표시하기 위해 첫 번째 위치(1행 2열의 첫 번째)에 서브플롯을 생성
    plt.plot(hist['train_acc'], label='Training acc') # hist 딕셔너리에서 학습 정확도(train_acc)를 추출하여 그래프로 그리는데 라벨을 'Training acc'로 지정하여 그래프에 범례를 추가
    plt.plot(hist['val_acc'], label='Validation acc') # hist 딕셔너리에서 검증 정확도(val_acc)를 추출하여 그래프로 그리는데 라벨을 'Validation acc'로 지정
    plt.legend() # 그래프에 범례를 추가하는데 각 데이터 세트를 구분하기 위해 사용
    plt.subplot(122) # 두 번째 위치(1행 2열의 두 번째)에 또 다른 서브플롯을 생성
    plt.plot(hist['train_loss'], label='Training loss') # hist 딕셔너리에서 학습 손실(train_loss)을 추출하여 그래프로 그린는데 라벨을 'Training loss'로 지정
    plt.plot(hist['val_loss'], label='Validation loss') # hist 딕셔너리에서 검증 손실(val_loss)을 추출하여 그래프로 그리는데 라벨을 'Validation loss'로 지정
    plt.legend() # 그래프에 범례를 추가

# 컨볼루션(Convolution) 연산을 시각화하는 함수 plot_convolution을 정의하는데 특정 커널을 사용하여 이미지에 적용한 결과를 보여줌 

def plot_convolution(t,title=''): # 함수를 정의하고, 두 개의 매개변수를 받는데 t: 컨볼루션 연산에 사용될 커널의 텐서이고 title: 그래프의 상단에 표시될 제목인데 기본값은 빈 문자열
    with torch.no_grad(): # 이 블록 내에서는 PyTorch의 자동 미분 기능을 비활성화하여, 연산에 대한 기울기 계산을 수행하지 않는데 메모리 사용을 줄이고 연산 속도를 향상
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1) # 3x3 크기의 커널을 사용하는 2D 컨볼루션 레이어를 생성하는데 입력 채널과 출력 채널이 모두 1
        c.weight.copy_(t) # 입력된 텐서 t를 컨볼루션 레이어의 가중치로 복사하는데 이렇게 설정하면 컨볼루션 연산이 정확히 이 가중치를 사용
        fig, ax = plt.subplots(2,6,figsize=(8,3)) # 2행 6열의 서브플롯을 생성하는데 그래프의 전체 크기는 가로 8인치, 세로 3인치
        fig.suptitle(title,fontsize=16) # 전체 그래프의 제목을 설정하는데 폰트 크기는 16
        for i in range(5): # 5번 반복하여 첫 번째 5개의 이미지에 대해 다음 작업을 수행
            im = data_train[i][0] # 학습 데이터셋에서 i번째 이미지를 호출
            ax[0][i].imshow(im[0]) # 첫 번째 행의 i번째 서브플롯에 원본 이미지를 표시
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0]) # 두 번째 행의 i번째 서브플롯에 컨볼루션 결과를 표시
            ax[0][i].axis('off') # 각 서브플롯의 축을 숨김
            ax[1][i].axis('off')
        ax[0,5].imshow(t) # 첫 번째 행의 6번째 서브플롯에 커널 텐서를 표시하는데 이 이미지는 컨볼루션 연산의 기본이 되는 커널을 시각화
        ax[0,5].axis('off') # 마지막 열의 축을 숨김
        ax[1,5].axis('off')
        #plt.tight_layout()
        plt.show() # 그래프를 표시

# 주어진 데이터셋에서 이미지를 선택하여 시각화하는 Python 함수 display_dataset을 정의하는데 이미지 데이터셋, 표시할 이미지의 수, 그리고 선택적으로 클래스 레이블을 포함할 수 있음

def display_dataset(dataset, n=10,classes=None): # display_dataset 함수를 정의하며, 매개변수로는 dataset (이미지와 레이블을 포함하는 데이터셋), n (표시할 이미지 수, 기본값은 10), classes (클래스 레이블 이름 배열, 선택적)를 받음
    fig,ax = plt.subplots(1,n,figsize=(15,3)) # 1행 n열의 서브플롯을 생성하고, 전체 그래프의 크기를 가로 15인치, 세로 3인치로 설정
    mn = min([dataset[i][0].min() for i in range(n)]) # 데이터셋에서 선택된 이미지들 중 픽셀 값의 최소값을 계산하는데 값은 이미지 정규화에 사용
    mx = max([dataset[i][0].max() for i in range(n)]) # 데이터셋에서 선택된 이미지들 중 픽셀 값의 최대값을 계산하는데 값은 이미지 정규화에 사용
    for i in range(n): # 0부터 n-1까지의 인덱스에 대해 반복
        ax[i].imshow(np.transpose((dataset[i][0]-mn)/(mx-mn),(1,2,0))) # 각 이미지를 정규화하고, 차원 순서를 변경하여 이미지를 표시하느데 PyTorch에서 일반적으로 사용되는 채널 첫 번째(C, H, W) 형식을 채널 마지막(H, W, C) 형식으로 변환
        ax[i].axis('off') # 각 서브플롯의 축을 숨김
        if classes: # classes 매개변수가 제공되었을 경우, 각 이미지 위에 해당하는 클래스 레이블을 제목으로 설정하는데 dataset[i][1]은 i번째 이미지의 클래스 인덱스
            ax[i].set_title(classes[dataset[i][1]])

# 주어진 파일 이름(fn)에 해당하는 이미지 파일을 검사하여 파일이 유효한 이미지인지 확인하는 Python 함수 check_image를 정의

def check_image(fn): # check_image라는 이름의 함수를 정의하며, 매개변수로 파일 이름 fn을 받음
    try: # 예외 처리를 시작하는 블록인데 안에서 이미지 파일을 열고 검증을 시도
        im = Image.open(fn) # Image.open(fn)을 사용하여 파일 fn을 열고, im 객체에 할당하는데 여기서 Image는 파이썬의 PIL(Pillow) 라이브러리에서 제공하는 모듈이고 이 함수는 이미지 파일을 불러오는 데 사용
        im.verify() # im.verify() 메소드를 호출하여 이미지 데이터가 손상되었는지 또는 파일이 손상된 이미지 포맷을 포함하고 있는지 검증하는데 파일을 읽을 때 발생할 수 있는 다양한 예외를 감지할 수 있고 이미지 파일이 유효하다면 이 부분은 문제없이 실행
        return True # try 블록이 성공적으로 완료되면, True를 반환하는데 파일이 유효한 이미지라는 것을 의미
    except: # try 블록에서 예외가 발생하면 실행되는 블록인데 예외가 발생하는 경우, 이미지 파일이 유효하지 않거나 파일 열기 과정에서 문제가 발생한 것
        return False # 예외가 발생했으므로, False를 반환하는데 이는 파일이 유효한 이미지가 아니라는 것을 나타냄

# 지정된 경로에 있는 모든 이미지 파일을 검사하고 손상된 이미지 파일을 찾아 삭제하는 Python 함수 check_image_dir를 정의

def check_image_dir(path): # check_image_dir라는 이름의 함수를 정의하며, 매개변수로 경로 패턴 path를 받는데 경로는 검색할 이미지 파일의 위치를 지정하는 글로브 패턴(glob pattern)을 포함할 수 있음
    for fn in glob.glob(path): # 함수를 사용하여 주어진 경로 패턴과 일치하는 모든 파일을 찾아 반복하는데 glob.glob은 지정된 패턴과 일치하는 모든 경로명을 리스트로 반환하는 함수
        if not check_image(fn): # check_image(fn) 함수를 호출하여 현재 파일 fn이 유효한 이미지인지 검사하는데 check_image 함수는 파일이 이미지로서 유효하지 않을 경우 False를 반환
            print("Corrupt image: {}".format(fn)) # 파일이 손상되었다면 해당 파일 이름과 함께 "Corrupt image" 메시지를 출력
            os.remove(fn) # os.remove(fn) 함수를 호출하여 손상된 이미지 파일을 파일 시스템에서 삭제

# PyTorch의 torchvision 라이브러리를 사용하여 이미지 변환을 위한 일반적인 변환 조합을 설정하는 함수 common_transform을 정의
# 전이 학습(Transfer Learning)이나 컴퓨터 비전 모델에서 이미지를 전처리할 때 매우 유용한데 이 변환을 사용하면 학습 데이터와 테스트 데이터를 모델이 기대하는 형식으로 일관되게 처리할 수 있음

def common_transform(): # common_transform이라는 이름의 함수를 정의하는데 이 함수는 매개변수를 받지 않고, 구성된 이미지 변환 파이프라인을 반환
    std_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]) # torchvision.transforms.Normalize 함수를 사용하여 이미지의 각 채널에 대한 정규화를 설정하는데 이 변환은 주어진 평균(mean)과 표준편차(std)를 사용하여 각 채널의 픽셀 값을 정규화하는데 이 값들은 일반적으로 ImageNet 데이터셋을 기준으로 한 통계값
    trans = torchvision.transforms.Compose([ # 여러 이미지 변환 단계를 조합하는 torchvision.transforms.Compose 함수를 사용
            torchvision.transforms.Resize(256), # 이미지의 크기를 256x256 픽셀로 조정
            torchvision.transforms.CenterCrop(224), # 이미지의 중앙을 기준으로 224x224 픽셀의 크기로 중앙을 자릅
            torchvision.transforms.ToTensor(), # 이미지 데이터를 PyTorch 텐서로 변환하고, 데이터 타입을 0에서 1 사이의 값으로 스케일링
            std_normalize]) # 정규화 변환을 적용
    return trans # 구성된 변환 파이프라인을 반환

# 개와 고양이의 이미지 데이터셋을 불러오고, 처리하여 학습 및 테스트 데이터셋으로 분할하는 Python 함수 load_cats_dogs_dataset를 정의하는데 함수는 데이터셋을 압축 해제하고, 이미지를 검사하며, 데이터를 분할하고, 로더를 설정하는 여러 단계로 구성

def load_cats_dogs_dataset(): # load_cats_dogs_dataset라는 이름의 함수를 정의합니다. 이 함수는 매개변수를 받지 않음
    if not os.path.exists('data/PetImages'): # 지정된 경로에 'PetImages' 폴더가 존재하는지 확인합니다. 폴더가 없으면 다음 단계로 이동
        with zipfile.ZipFile('data/kagglecatsanddogs_5340.zip', 'r') as zip_ref: # 'kagglecatsanddogs_5340.zip'라는 이름의 압축 파일을 읽기 모드로 열고 zip_ref 객체로 참조
            zip_ref.extractall('data') # zip_ref 객체를 사용하여 압축 파일 내의 모든 내용을 'data' 디렉토리에 압축 해제

    check_image_dir('data/PetImages/Cat/*.jpg') # 'data/PetImages/Cat' 폴더 내의 모든 '.jpg' 파일을 검사하여 손상된 이미지가 있는지 확인하고, 손상된 이미지는 삭제
    check_image_dir('data/PetImages/Dog/*.jpg') # 'data/PetImages/Dog' 폴더 내의 모든 '.jpg' 파일도 동일하게 검사

    dataset = torchvision.datasets.ImageFolder('data/PetImages',transform=common_transform()) # ImageFolder 클래스를 사용하여 'data/PetImages' 디렉토리의 이미지들을 로드하고 common_transform() 함수를 호출하여 이미지에 적용할 변환을 설정
    trainset, testset = torch.utils.data.random_split(dataset,[20000,len(dataset)-20000]) # 데이터셋을 무작위로 20,000개의 학습 셋과 나머지를 테스트 셋으로 분할
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32) # 학습 데이터셋에 대한 데이터 로더를 생성하고, 배치 크기를 32로 설정
    testloader = torch.utils.data.DataLoader(testset,batch_size=32) # 테스트 데이터셋에 대한 데이터 로더를 생성하고, 배치 크기를 32로 설정
    return dataset, trainloader, testloader # 완성된 데이터셋과 데이터 로더들을 반환
