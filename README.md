# Native-APP
## 교통위반 감지 모델

이 프로그램은 교통위반 감지 모델을 이용한 Native-APP으로 PyQt5를 사용하여 GUI를 만들고, OpenCV와 PyTorch를 활용하여 실시간 객체 검출을 수행하는 어플리케이션입니다.

GUI 디자인: UI는 Qt Designer를 통해 디자인되었습니다. input QLabel은 입력 이미지를 표시하고, output QLabel은 객체 검출 결과를 표시합니다. 그리고 label, label_2, label_3는 각각 검출된 객체의 클래스와 신뢰도를 표시합니다. 마지막으로 pushButton은 이미지 파일을 불러오는 버튼입니다.

이벤트 처리: 불러오기 버튼(pushButton)을 클릭하면 파일 불러오기 다이얼로그가 열리고, 선택한 이미지가 처리됩니다. 이 때, load_image 함수가 호출되어 이미지를 불러와 처리합니다.

실시간 객체 검출: video_pred 함수에서는 웹캠에서 프레임을 읽어와 YOLO 모델을 사용하여 객체를 탐지하고, 그 결과를 GUI에 표시합니다. 만약 파일을 불러왔다면, 이 함수는 불러온 이미지를 사용하여 객체 탐지를 수행합니다.

시간 측정: 객체 탐지 시간을 측정하여 GUI에 표시합니다.

이 어플리케이션을 실행하면, GUI를 통해 실시간으로 웹캠의 영상을 확인하고, 객체가 검출되면 그 결과를 화면에 표시할 수 있습니다. 또한 파일을 불러와서도 객체 탐지를 수행할 수 있습니다.
