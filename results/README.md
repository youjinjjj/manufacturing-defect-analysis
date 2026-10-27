# Historical results

이 폴더의 그림과 수치는 기존 과제 노트북에 저장된 출력에서 가져왔습니다. 게시 과정의 새로운 실험 결과가 아닙니다.

- `historical_classifier_comparison.csv`: 원 Final_Project의 5-fold 교차검증 요약(원본 cell 67)
- `historical_test_confusion_matrix.csv`: 원 Final_Project의 XGBoost 시험셋 혼동행렬(원본 cell 70)
- `figures/`: 원 노트북의 선택된 PNG 출력, 이미지 바이트 변경 없음

교차검증 지표와 시험셋 지표는 서로 다른 데이터 분할에서 계산되므로 직접 같은 수치로 비교하지 않습니다.
`class=1`은 정상이며 기본 F1·precision·recall도 정상 기준입니다.

## Defect-oriented interpretation of the saved test result

| 항목 | 값 |
|---|---:|
| 시험셋 불량 수 | 14 |
| 탐지된 불량 | 6 |
| 누락된 불량 | 8 |
| 불량 precision | 6 / (6 + 2) = 0.7500 |
| 불량 recall | 6 / (6 + 8) = 0.4286 |
| 불량 F1 | 12 / (12 + 2 + 8) = 0.5455 |

위 비율은 저장된 혼동행렬에서 산술적으로 다시 계산한 해석이며, 모델을 재학습한 수치가 아닙니다.
