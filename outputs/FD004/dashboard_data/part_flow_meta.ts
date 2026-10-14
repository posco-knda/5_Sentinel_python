// ⚠️ 자동 생성 파일 — 분석 레포의 scripts/20_export_part_flow.py 가 만든다. 직접 고치지 말고 파이썬을 다시 돌려서 받아올 것.
// 공식 test 엔진의 센서 15개가 초기 상태(처음 30사이클)에서 얼마나 벗어났는지(이탈도 z)와 부품별 요약 (생성: 2026-10-14T14:27:57).
// 고장 원인 판정이 아니라 "가장 크게 변한 센서가 속한 부품(추정)"이다 — 분석 레포 scripts/20_export_part_flow.py 상단 설명 참고.

export interface PartMeta {
  id: string
  label: string
  sensors: { id: string; name: string; unit: string }[]
}

export const partMeta: {
  baselineCycles: number
  recentCycles: number
  parts: PartMeta[]
  sigma: Record<string, number>
  redTopPart: Record<string, number>
  redEngines: number
  profile: Record<'red' | 'warn' | 'ok', Record<string, number>>
  red: number
  yellow: number
} = {"baselineCycles":30,"recentCycles":15,"parts":[{"id":"fan","label":"팬·바이패스","sensors":[{"id":"s8","name":"Nf · 물리적 팬 속도","unit":"rpm"},{"id":"s13","name":"NRf · 보정 팬 속도","unit":"rpm"},{"id":"s15","name":"BPR · 바이패스비","unit":""},{"id":"s6","name":"P15 · bypass-duct 전압력","unit":"psia"}]},{"id":"lpc","label":"저압압축기(LPC)","sensors":[{"id":"s2","name":"T24 · LPC(저압압축기) outlet 온도","unit":"°C"}]},{"id":"hpc","label":"고압압축기(HPC)","sensors":[{"id":"s3","name":"T30 · HPC(고압압축기) outlet 온도","unit":"°C"},{"id":"s7","name":"P30 · HPC outlet 전압력","unit":"psia"},{"id":"s11","name":"Ps30 · HPC outlet 정압","unit":"psia"}]},{"id":"core","label":"코어·연소·블리드","sensors":[{"id":"s9","name":"Nc · 물리적 코어 속도","unit":"rpm"},{"id":"s14","name":"NRc · 보정 코어 속도","unit":"rpm"},{"id":"s12","name":"phi · 연료 유량/Ps30 비","unit":"pps/psi"},{"id":"s17","name":"htBleed · 블리드 엔탈피","unit":""}]},{"id":"hpt","label":"고압터빈(HPT)","sensors":[{"id":"s20","name":"W31 · HPT(고압터빈) 냉각 블리드 유량","unit":"lbm/s"}]},{"id":"lpt","label":"저압터빈(LPT)","sensors":[{"id":"s4","name":"T50 · LPT(저압터빈) outlet 온도","unit":"°C"},{"id":"s21","name":"W32 · LPT(저압터빈) 냉각 블리드 유량","unit":"lbm/s"}]}],"sigma":{"s8":0.061,"s13":0.064,"s15":0.021,"s6":0.006,"s2":0.305,"s3":3.87,"s7":0.528,"s11":0.105,"s9":4.266,"s14":3.382,"s12":0.434,"s17":0.945,"s20":0.102,"s4":4.062,"s21":0.062},"redTopPart":{"core":39,"hpc":15,"fan":2,"lpt":3,"lpc":3},"redEngines":62,"profile":{"red":{"fan":3.7,"lpc":2.1,"hpc":3.9,"core":5.3,"hpt":1.8,"lpt":2.7,"n":62},"warn":{"fan":2.2,"lpc":1.0,"hpc":2.0,"core":2.9,"hpt":0.8,"lpt":1.3,"n":50},"ok":{"fan":0.9,"lpc":0.4,"hpc":0.9,"core":1.2,"hpt":0.4,"lpt":0.5,"n":119}},"red":40,"yellow":80}
