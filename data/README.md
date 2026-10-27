# Dataset preparation

원 과제에서는 **KAMP 사출성형 AI 데이터셋**의 `labeled_data.csv`를 사용했습니다.
원 데이터 제공처: [KAMP 제조 AI 데이터셋 플랫폼](https://www.kamp-ai.kr/).
사이트에서 사출성형 데이터셋의 설명과 이용 조건을 확인하고 원 과제와 같은 labeled 데이터를 준비하세요.
데이터 제공처의 이용 절차가 필요할 수 있습니다.

```text
data/
├── README.md
└── labeled_data.csv
```

기존 노트북에 기록된 입력은 **7,996행, 45열**이며 `PassOrFail`은 `Y`(정상)와 `N`(불량)입니다.
다른 버전의 파일은 행 수·스키마·결과가 다를 수 있습니다.
사용하는 주요 열은 다음과 같습니다.

- `PassOrFail`, `Reason`, `PART_NAME`, `EQUIP_CD`, `EQUIP_NAME`
- `Injection_Time`, `Filling_Time`, `Cycle_Time`
- `Max_Injection_Speed`, `Average_Screw_RPM`, `Max_Injection_Pressure`
- `Barrel_Temperature_*`, `Mold_Temperature_*`, `Hopper_Temperature`

데이터 원본은 저장소에 재배포하지 않습니다. 이용 조건은 원 제공처를 따릅니다.
원 수업에서 받은 파일이 있다면 `data/labeled_data.csv`에 놓으면 됩니다.
