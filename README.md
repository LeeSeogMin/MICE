# 디지털 전환의 생산성 효과: MICE-이중강건 인과추론 연구

**한국 제조업 데이터를 활용한 디지털 전환의 총요소생산성(TFP) 효과 분석**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-completed-green)](https://github.com/LeeSeogMin/MICE)

---

## 🎯 프로젝트 개요

본 연구는 **Multiple Imputation by Chained Equations (MICE)**와 **이중강건(Doubly Robust) 추정량**을 결합하여 디지털 전환이 한국 제조업체의 총요소생산성에 미치는 인과효과를 추정합니다.

### 연구 질문

> **디지털 전환(IoT, Cloud, AI, Big Data)은 제조업체의 생산성을 향상시키는가?**

### 핵심 기여

1. **결측치 처리**: MICE를 통한 표본 크기 76% 확대 (17,897 → 31,572)
2. **인과추론**: AIPW를 통한 이중강건 추정 (처치 불균형 10.7:1 극복)
3. **강건성 검증**: Placebo 검정(1000회), 하위표본 분석(COVID, 산업별)
4. **재현성**: 완전히 재현 가능한 파이프라인 (고정 시드, 상세 문서화)

### 주요 발견

- **평균 처치효과 (ATE)**: +3.9% (p<0.001, 95% CI=[2.1%, 5.7%])
- **산업별 이질성**: 화학(+5.4%), 전자부품(+4.7%), 기계(+1.1%)
- **COVID 효과**: Pre(+1.4%) → During(-0.7%) → Post(+0.2%)

---

## 📂 프로젝트 구조

```
MICE/
├── README.md
├── requirements.txt
│
├── analysis/
│   ├── 00_imputation.py               # MICE 다중대체
│   ├── 02_tfp_levinsohn_petrin.py     # TFP 추정
│   ├── 03_doubly_robust_aipw.py       # AIPW 이중강건 추정 (메인)
│   ├── 04_tmle.py                     # TMLE 비교
│   ├── 19_placebo_tests.py            # Placebo 검정
│   ├── run_additional_analyses.py     # COVID/산업별 이질성
│   ├── processed_data/                # 처리된 데이터
│   └── results/                       # 분석 결과
│
└── raw data/                          # 원본 데이터 (Statistics Korea)
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/LeeSeogMin/MICE.git
cd MICE

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 데이터 준비 (Statistics Korea 승인 필요)

```bash
# 1. MDIS(https://mdis.kostat.go.kr)
# 2. 2019-2023 기업활동조사 다운로드
# 3. raw data/ 디렉토리에 배치
#    파일명 예시: 2019_기업활동조사_20200115_12345.csv
```

### 3. 전체 파이프라인 실행

```bash
# 단계 1: MICE 다중대체
python analysis/00_imputation.py

# 단계 2: TFP 추정
python analysis/02_tfp_levinsohn_petrin.py

# 단계 3: AIPW 이중강건 추정 (메인 결과)
python analysis/03_doubly_robust_aipw.py

# 단계 4: TMLE 비교 (선택사항)
python analysis/04_tmle.py

# 단계 5: Placebo 검정 (선택사항)
python analysis/19_placebo_tests.py

# 단계 6: 이질성 분석 (선택사항)
python analysis/run_additional_analyses.py
```

### 4. 결과 확인

```powershell
# 메인 결과
type analysis\results\aipw_report.txt

# 추가 분석 결과 (선택사항)
type analysis\results\covid_period_analysis.csv
type analysis\results\industry_heterogeneity.csv
```

---

## 📊 주요 결과

### 평균 처치효과 (ATE)

| 방법          | ATE         | SE     | 95% CI           | 효과          |
| ------------- | ----------- | ------ | ---------------- | ------------- |
| **MICE-AIPW** | **+0.0391** | 0.0091 | [0.0212, 0.0569] | **+3.98%** ✅ |

> 디지털 전환은 제조업 총요소생산성을 평균 **3.98%** 향상시킵니다.

### COVID 기간별 효과

| 기간                     | N      | ATE     | 효과   |
| ------------------------ | ------ | ------- | ------ |
| Pre-COVID (2019)         | 6,330  | +0.0138 | +1.39% |
| During-COVID (2020-2021) | 12,300 | -0.0068 | -0.68% |
| Post-COVID (2022-2023)   | 12,942 | +0.0023 | +0.23% |

### 산업별 이질성

| 산업 (KSIC)       | N     | ATE         | 효과          |
| ----------------- | ----- | ----------- | ------------- |
| **20 (화학)**     | 2,394 | **+0.0530** | **+5.44%** ⭐ |
| **26 (전자부품)** | 3,378 | **+0.0454** | **+4.65%** ⭐ |
| 30 (기타운송)     | 3,644 | +0.0446     | +4.56%        |
| 29 (기계)         | 4,126 | +0.0110     | +1.10%        |
| 10 (식품)         | 2,330 | +0.0013     | +0.13%        |

---

## 🔬 방법론

### 데이터

**출처**: 한국 기업활동조사 (2019-2023, Statistics Korea)

- 원본: 84,000+ 기업-연도 관측치
- 처리 후: 31,572 (MICE 대체 후)
- 공변량: 281개 (재무, 산업, 지역, R&D 등)

### 처치 변수

**디지털 전환 지수 (DTI)**:

- IoT, Cloud, AI, Big Data의 평균 채택 수준
- 이진화: DTI > 0 → 채택, DTI = 0 → 비채택
- 처치율: 8.5% (10.7:1 불균형)

---

## 🛠️ 시스템 요구사항

- **Python**: 3.9+
- **RAM**: 8GB 이상 권장
- **의존성**: numpy, pandas, scikit-learn, scipy, statsmodels

---

## 데이터

**접근**: [MDIS Statistics Korea](https://mdis.kostat.go.kr)

---
