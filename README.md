data_tools
data_saparate.py는 한 번만 실행합니다.(train, val, test 비율을 8:1:1로 )
filter_dataset.py 는 데이터에 이미지와 라벨링 데이터가 맞지 않아 다시 돌려서 안 맞는 녀석들은 제거
image_label_matching.py로 매칭여부 확인
json2yolo 욜로 돌리기 위해서는 json 파일을 txt로 전환해야 합니다. 한번만 돌리고 json 파일은 따로 빼놨습니다(label_data(json))
