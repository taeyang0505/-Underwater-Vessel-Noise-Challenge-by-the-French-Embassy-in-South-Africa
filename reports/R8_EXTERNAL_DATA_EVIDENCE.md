# 외부 데이터 원천·EDA·전처리 증거 inventory (R8)

이 문서는 **다운로드→고유 recording→변환 bank→과거 CV→현재 참조 pool**을 구분한다. 숫자는 실제 보존 metadata에서 다시 계산했으며, waveform/FLAC/band array를 열거나 새 수집·복원·학습을 하지 않았다. 이는 완료된 R2/R4/R5/R6/R7 버전의 inventory다. 이후 새로 승인된 R9 전체 참조 pool은 별도 새 manifest로 기록하며 이 과거 편입 수를 소급 변경하지 않는다. 모든 입력 파일SHA와 원시 schema/전체 class/split/count는 [기계 판독 증거](R8_EXTERNAL_DATA_EVIDENCE.json)에 있다.

## 1. 현재와 과거를 구분한 크기

| 분모/버전 | 고유 clip/crop | 합산 시간 | 현재 상태 |
|---|---:|---:|---|
| VTUAD strict 전체변환 | 525=509vessel archive+16ambient | 42.046667h | strict만다운;12classcontainer/logmel미사용 |
| HMS WAV 고유 bytes | 1,145(다운로드entry1,146) | 6.382677h | 모두10–30s,measuredcomplete-minute0 |
| 과거R2CV VTUAD | 218 | 14.315000h | 과거편입;reserved/excluded유지 |
| 과거R2CV HMS | 945 | 5.270563h | 19daycomponents;현재R4/R5/R6에는0 |
| NOAA expansion primary | 12×15min | 3.00h | oldCV8=2h,cal2=.5h,sealed2=.5h |
| NOAA PMNoShip quarantine | 1×15min | .25h | positive·자연선박부재GT둘다아님 |
| 별도oldNOAA OC02평가 | 3×15min | .75h | 이미평가됨,fit/calibration/modelselection미사용 |
| R4/R5선택VT 전체5fold pool | 119/22components | 8.631944h | Fold0 TRAIN69/12,INNER25/5,oldOUTER25/5 |
| 현재R6guardrailsVT 전체역할 | 94/17components | 7.048889h | oldOUTER25/5제외;아래역할별구분 |

모든 시간은 **clip header 또는900s crop의 단순 합**이다. 완전한2s bank의 실제사용초와완전minute초도JSON에따로기록했다(R4선택header31,075s/완전2s사용31,070s,R6header25,376s/사용25,372s). 겹치는 날짜·같은 선박·원 recording support를 정리한 독립 현장녹음 시간이 아니다. 원래 변환 bank는525+1145=1670,NOAAprimary12추가 inventory는1682(별도PMquarantine제외),과거CV는218+945+8=1171/26components다. 현재R4/R5/R6를1171개외부clip으로 학습했다고 읽으면 안 된다.

| 현재R6역할 | clips | components | 합산header h |
| --- | --- | --- | --- |
| cross_generator | 18 | 3 | 1.150000 |
| cross_sensor | 7 | 2 | 0.433333 |
| inner | 18 | 3 | 1.332778 |
| train | 51 | 9 | 4.132778 |

## 2. VTUAD rebuilt v1.3.2

원천은 ONC hydrophone/AIS 기반의 자동 scenario label이다. [공식VTUADrecord](https://zenodo.org/records/23037895)는 CC BY4.0이며, strict와12per-class container는 같은525scenario의 다른cut view다. [ONC정책](https://www.oceannetworks.ca/data/data-policy/)은 ONC단독소유와partner제한자료를 구분한다. localstrictheader42.046667h와공개~56h의 차이는 전체대응header대조가없어 정확원인을확정하지 않았다.

다운로드WAVvolumes4개는vessel.part01–03+strictbackgroundZIP로4,638,375,330bytes다. providerMD5검증기록과currentfilesize가일치하며525WAV의per-WAVSHA검증이변환manifest에기록돼있다. 이번감사에서4.6GB를재hash하거나오디오를재decode하지않았다. 별도metadata.csv대신registry_anonymized.csv22columns를사용했다. [직전archive대조](R7_ANIL45_SOURCE_COMPARISON.md).

실측metadata: **32kHz/mono525**,길이119–2220s,중앙값180s,q75=299s,완전2sbin75,678개/완전minute2,511개,33recordingdays다. 아래는525분모의rawweaklabel이다.

| VT raw weak class | clips/525 |
| --- | --- |
| tug | 188 |
| cargo | 97 |
| passengership | 71 |
| other | 40 |
| background | 38 |
| fishing | 34 |
| pleasurecraft | 32 |
| dredger | 8 |
| sailing | 6 |
| tanker | 5 |
| rescue | 4 |
| pilotvessel | 2 |

38background label은dedicatedambient16+vesselarchive-role안의weak-background22다. archive_role과class를동일시하지않는다. registry의hull_id16결측은ambient와대응하며CTD계열10columns는각10/525결측이다. 이covariate/ID를현재acousticgrid의정답으로대체하지않았다.

**선택과품질:**525→원TRAIN/validation243→positive자격218→bridge99제외로119/22→oldOUTER25/5제외로R6 94/17이다. 307미편입은cal6+domain15+hull/dayconflict260+crossday1+자격없는train/val25(ambient10+vessel-role비positive15)로구분한다. 이를모두background라표현하거나자동복원하지않는다. 119/22는선택pool의직접격리이고전체525보수graph는1component다. 새hydrophone22개가생긴것이아니다.

QCmetadata는525모두verified/measured/full-band검사를기록하면서 `large_DC_fraction_review`를남긴다. periodogram의constantdetrend는평균을제거할뿐,압력calibration이나모든센서특성문제의해소를뜻하지않는다. AIS구간과class는순수vessel-only파워의cellGT가아니다. dedicatedambient16은변환·보관했지만현재R4/R5/R6positive참조에는0개다.

## 3. HearMyShip: 짧은CPA원천

[Scientific Data2025 원논문](https://www.nature.com/articles/s41597-025-04584-x)은크로아티아Šibenik의음향+영상관측과QC를설명한다. [ScienceDBdeposit](https://doi.org/10.57760/sciencedb.13099)의보존license는CC0-1.0이다. 원논문/설명1148transits와현재다운로드1146WAVentry의차이는이감사에서완전히대응시키지않았고1148독립hull로세지않는다.

보존entry1146개가모두MD5verified이며하나의Yacht파일이같은MD5로2회등장해고유WAV는1145개다. 중복entry하나는별도SHA필드를반복저장하지않지만같은localpath를가리키고다른entry/변환manifest에고유SHA가있다. uniqueWAVbytes는3,315,250,540다. rawschema는name/URL/MD5/size/time/category/length/speed/localpath/SHA이며1146분모에서mirror_id·length·speed각9결측이다. 예전acquisitionplan의초기9미매칭을최종9미다운로드로읽지않는다.

실측header는**48kHz/mono1145**,길이10–30s/중앙20s,합6.382677h이고완전minute는0이다. 과거CV945(train767+validation178)만5.270563h이며,cal144/domain56은제외됐다. recurringhull은검증되지않아daygroup19를19독립선박으로부르지않는다. 현재R4/R5/R6참조에는HMS0개다.

| HMSrawcategory(별칭보존) | 고유clips/1145 |
| --- | --- |
| Motorboat | 709 |
| Sailboat | 191 |
| Yacht | 123 |
| Tour boat | 59 |
| Ferry | 51 |
| TourBoat | 5 |
| Fishing Vessel | 2 |
| Cargo Ship | 2 |
| Tourboat | 2 |
| Auxiliary ship | 1 |

Tour boat/TourBoat/Tourboat는원래문자열별통계다. 이를물리적으로서로다른선종이라고주장하지않는다. Motorboat709/1145로편중돼있다. 소스원논문의calibratedrecording시스템과달리우리변환기는sensitivity/gain/propagationcalibration을적용하지않아WAVdigitalpower를pressure단위로해석하지않는다.10–30s를반복해서측정15분으로명명하지않는다.

## 4. NOAA SanctSound: 연속900s발췌와예약

[원시음향공식80342](https://www.fisheries.noaa.gov/inport/item/80342),[주석·품질products80343](https://www.fisheries.noaa.gov/inport/item/80343)의현행AccessInformation은둘다CC0-1.0이다. 원본FLAC/WAVobject와generation,annotation시간구간,Goodfrequency구간을join한뒤native900×fs프레임만확보했다. station/deployment/device/UTC/frame/header/checksum이rawschema에남는다.

| NOAAexpansion예약/역할 | clips | crop h |
| --- | --- | --- |
| ambient_control_quarantine | 1 | 0.25 |
| calibration | 2 | 0.50 |
| new_sealed_holdout | 2 | 0.50 |
| train | 6 | 1.50 |
| validation | 2 | 0.50 |

primary12는6deployment/12days이며fs48k10개+96k2개,모두mono/900s다. oldCV8clip은CI/FK/GR의train6+HIvalidation2(4sourcecomponents)였다.OC03cal2와SBsealed2는CV에서제외했다. PMNoShip1은별도quarantine이며자연선박부재의정답으로쓰지않는다. 별도OC02_02old3crop(.75h)은이미평가해fit/보정/반복선택에쓰지않는다. 이문서는그들의metadata만읽었다. **현재R4/R5/R6모든NOAApositive참조는0**이다.

sourceQC는정확한shipannotation∩전체25bandGood구간∩실제native연속900s,1초이상digitalzerogap없음·rail≤.1%조건이었다. primary12의기록railfraction은모두0이다. 일부개별zero샘플이있다는사실은긴gap검증실패와다르다. OC01의all-bandCompromised,MB01의저역경고,FK의seafloorcontact기간등은구간별로제외했고전체NOAA가저품질이라는판정은하지않았다.

무결성은**generation고정HTTP206range/Content-Range/범위SHA/발췌SHA/변환SHA**에한정한다. provider전체objectMD5는재계산하지않았고거대원본전체를내려받았다고하지않는다. 센서가공식적으로calibrated였어도우리발췌digitalpower에교정계수를적용하지않았다. Ship구간은다른ambient가포함된recording이지순수선박energy나독립hull확인이아니다.

## 5. 공통25band처리와소스별처리의경계

1. 각소스먼저파일권리·providerchecksum·nativeheader·sourcegroup/role·개별QC를확인한다. 다른source의공식라벨을추가에너지share와동일시하지않는다.
2. actualnativefs/channel0로읽고완전한2sHann/detrendconstant/densityPSD를구한다. df=.5Hz,centre `10^(1.5+.1j)`25개에base10edge `[c/10^.05,c*10^.05)`의PSD×df를적분한다.
3. centre31.62–7943.28Hz,실제상단8912.51Hz를지원해야한다. nativefs16k를upsample해전25band가측정됐다고표기하지않는다.
4. 완전30개2sbin의**선형파워평균**만측정1분이다.2s미만tail은버리되완전2sbin은shortbank에남긴다. dB평균이아니다. HMS는minute0이면서2sbin은보존한다.
5. 출력은**uncalibrateddigitalmean-squarepower**다. WAV형식/샘플레이트변환만으로서로다른센서의응답과절대감도가같아지지않는다.
6. R4/R5는2sbank의boundedpowerwarp/fade/EQ뒤minute평균을했고R6는native-minute-first/realshortcrop,no warp,EQ6/nolowpass를쓴다. 변형길이를원UTC·measuredlength에덮어쓰지않는다.
7. 합성에서아는것은 `E=B+A, A=alpha·R, q=A/E, y=sum(OA)/sum(OE)`다. R에녹음ambient도포함돼**controlledadded-recordingfraction**을학습하며원래SA관측에대한naturalshipenergy정답을발견한것이아니다. 원래label·원래관측·합성GT의세층위를보존한다.

## 6. 탐색만했거나metadata만받은자료

| 후보 | 공개원천확인/접근 | 로컬audio/현재사용 | 판단 |
|---|---|---|---|
| [DeepShipofficialrepo](https://github.com/irfankamboh/DeepShip) | README47h04/265ships/4classes,일부공개·나머지email요청 | 0/0 | 명시적재사용data licence를현재페이지에서확정하지못함;CC라고가정하지않음 |
| [ShipsEarofficial](https://underwaternoise.atlanttic.uvigo.es/) | 샘플링크·전체access요청;환경잡음도포함 | 0/0 | dataset권리/실제header/구간미감사;논문license를data권리로대체하지않음 |
| [ONCdirect](https://www.oceannetworks.ca/data/data-policy/) | ONC단독소유CCBY4/partner조건별도 | 별도신규directaudio0 | VTUADupstream과중복기원;독립추가dataset으로중복계산하지않음 |
| PORTIA | 4metadataarchiveMD5확인 | audio0/current0 | 외부metadata95,022shortwindows는원음연속15분보장아님 |
| VLIZ/MBARI/Vulcano | 공식catalogue·권리·접근문서조사 | audio0/current0 | nativefs/시간연속성/권리·label단위추가검증전미편입 |

이상후보에연락·계정행동·새다운로드를하지않았다. 검색상관심/공개download가능성/우리실제수집/현재사용은각각다르다. 수집하지않았다는사실을음질불량판정으로바꾸지않는다. [보존후보catalogue](external_expansion_catalog.json).

[전체숫자·분모·schema·QC·SHA](R8_EXTERNAL_DATA_EVIDENCE.json) · [아카이브비교](R7_ANIL45_SOURCE_COMPARISON.md)
