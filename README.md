# Dacon - AI 양재 허브 인공지능 오픈소스 경진대회 (Image Super-Resolution)
## 팀정보
팀명 : 가보즈아~ 

구성원 : DerekIm, NyongKim

성적 : Public, Private 3등

## 개발 환경
OS : ubuntu 18.04.5 LTS

CPU : Inter(R) Xeon(R) Gold 6130 CPU @ 2.10GHz

GPU : V100 * 4

## 데이터셋 구성
```bash
├── datasets
│   ├── train
│   │   ├── lr
│   │   ├── hr
│   ├── test
│   │   ├── lr
```

## 코드 설명
```
./data 폴더 : dataset 관련 코드
./model 폴더 : model 관련 코드
./options 폴더 : train.json / test.json으로 구성 --> 모델 구조 및 하이퍼파라미터, GPU ID등을 설정
./utils 폴더 : 각종 utility 관련 코드
dataConvert.py : 학습 데이터셋 전처리 관련 코드
daconTrain.py : 모델 학습 코드
daconTest.py : 모델 테스트 코드
```



## Code 실행 순서

1. 코드 다운로드
```
git clone https://github.com/seokjuice/daconSR.git
cd daconSR
```

2. 라이브러리 다운로드
```
pip install -r requirement.txt
```

3. 데이터 전처리 (코드 내 original dataset 및 save path 설정 필수 / original dataset path는 train, test 폴더 상위폴더경로로 설정 / ex) "./datasets/"
```
python dataConvert.py 
```

4. 모델 학습 
1) 커맨드 형식
```
python -m torch.distributed.launch --nproc_per_node=(number of gpus) --master_port=1234 daconTrain.py --opt options/train.json  --dist True --dataPath_lr (root for low resolution images) --dataPath_hr (root for high resolution images)
```
2) 예시 (이때, train.json 파일 내 gpu_ids 설정 필수)
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 daconTrain.py --opt options/train.json  --dist True --dataPath_lr "/home/work/daconSR/datasets/32interval_128/lr/"  --dataPath_hr "/home/work/daconSR/datasets/32interval_128/hr/"
```

5. 모델 테스트
1) 커맨드 형식
```
python daconTest.py --weightPath (Directory path containing model weights) --dataPath (Directory path containing test image) --savePath (Save path) --modelVersion (weights name except _E and _G)
```
2) 예시 (이때, train.json 파일 내 gpu_ids 설정 필수)
```
python daconTest.py --weightPath "./daconSR/psnr25.89/" --dataPath "./testImages/lr/" --savePath "./psnr25_89" --modelVersion model1 model2 model3 model4
```

## 모델 테스트 결과
modelVersion에 한개의 모델 이름을 입력하면 복원 결과 및 submission.zip 

복수개의 모델명이 입력된 경우 각 모델의 복원 결과와 앙상블한 이미지, 최종 submission.zip 생성






