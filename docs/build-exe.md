# 단일 실행 파일(.exe) 만들기

Windows에서 Python 없이 더블클릭으로 실행되는 `WarpageAnalysis.exe` 하나로
묶는 방법. 도구는 **PyInstaller**.

## 1. 준비 (한 번만)

빌드할 PC에 프로그램이 정상 실행되는 Python 환경이 먼저 있어야 함.

```bash
# 저장소 루트에서
pip install -r requirements.txt
pip install dash==2.17.1 plotly==5.24.1 kaleido==0.2.1   # requirements에 없으면
pip install pyinstaller
```

> kaleido는 PNG(이미지) 저장에 필요. 없으면 "Save ... as PNG" 기능만 안 됨.

## 2. 빌드

**저장소 루트**에서 (packaging 폴더 안 아님):

```bash
pyinstaller packaging/warpage.spec
```

끝나면:

```
dist/WarpageAnalysis.exe   ← 이 파일 하나가 전부
```

## 3. 실행

`WarpageAnalysis.exe` 더블클릭 →
- 검은 콘솔 창이 뜸 (로그·오류 표시용, 닫지 말 것)
- 1~2초 뒤 기본 브라우저에서 `http://127.0.0.1:8050` 자동으로 열림
- 종료: 콘솔 창에서 `Ctrl + C` 또는 창 닫기

다른 PC로 옮길 때 `.exe` 파일만 복사하면 됨. Python 설치 불필요.

## 왜 그냥 `pyinstaller app_main.py`가 아니라 .spec인가

Dash·plotly·kaleido는 코드 외에 **데이터 파일**(JS 번들, plotly validators,
kaleido 렌더러 실행파일)을 함께 씀. PyInstaller 기본 분석은 이걸 못 챙김 →
`warpage.spec`의 `collect_all()`이 모아줌. 앱 자체 CSS(`assets/style.css`)도
spec에서 직접 추가함.

## 옵션 조정 (`packaging/warpage.spec`)

| 원하는 것 | 수정 |
|---|---|
| 아이콘 넣기 | `icon="packaging/app.ico"` 주석 해제 후 .ico 준비 |
| 콘솔 창 숨기기 | `console=False` — 단, 오류가 안 보이니 배포 전 충분히 테스트 |
| exe 용량 줄이기 | `demo_data` 포함 줄(`datas += [("demo_data"...)]`) 주석 처리 |
| 실행 폴더형(빠른 시작) | `onefile=True` → `False`. `dist/WarpageAnalysis/` 폴더 통째 배포 |

## 흔한 문제

- **`ModuleNotFoundError` (실행 시)** — 누락된 패키지를 spec의 `collect_all`
  루프 또는 `hiddenimports`에 추가.
- **브라우저는 열리는데 화면이 비거나 CSS가 안 먹음** — assets 경로 문제.
  spec의 `datas += [("src/matrix2d/ui/assets", "matrix2d/ui/assets")]` 확인.
- **PNG 저장 실패** — kaleido 미포함. `pip install kaleido==0.2.1` 후 재빌드.
- **onefile exe가 느리게 시작** — 정상. 매 실행마다 임시폴더로 압축을 풂.
  빠른 시작이 필요하면 위 "실행 폴더형"으로.
- **백신이 오탐** — PyInstaller onefile의 알려진 현상. 서명하거나
  폴더형 배포로 완화.
```
