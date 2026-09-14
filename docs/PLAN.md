# Roadmapa projektu

Dokument roboczy. README jest po angielsku, bo repo jest publiczne; ten plik
jest nasz.

## Cel

Uczciwe sformułowanie, do którego wracamy przy każdej decyzji:

> Sprawdzamy, które obwody muszej sensomotoryki przenoszą się do obcego ciała,
> a które się rozsypują i dlaczego.

Czego **nie** robimy: nie "wgrywamy muchy", nie budujemy sztucznej świadomości,
nie dostrajamy odczytu aż robot zacznie chodzić.

## Podział pracy

| Ja | Ty |
|---|---|
| pipeline danych, silnik LIF, integracje, protokół mózg-ciało | zakres, budżet, decyzje |
| harness eksperymentalny i baseline'y | sprzęt: zakup, montaż, kalibracja |
| kod etapów 1-4 | weryfikacja, czy wyniki mają sens |

## Etapy

### Etap 1 - mózg na biurku (ZROBIONE)

- [x] kontener na konektom (`Connectome`) + budowa macierzy ze znakiem
- [x] tabela neuroprzekaźnik -> znak, z glutaminianem jako **hamującym**
- [x] silnik LIF na parametrach Shiu et al., event-driven
- [x] `peak_psp_factor` / `synapses_to_threshold` - ~162 synapsy na jeden spike
- [x] syntetyczny fixture do testów offline
- [x] loadery MaleCNS i FlyWire
- [ ] **uruchomić na realnych danych** - wymaga sieci, patrz `docs/DATA.md`
- [ ] test odbiorczy na realnym konektomie: LC4 -> GF -> DNp09/TTMn

### Etap 2 - oczy (ZROBIONE)

- [x] heksagonalna siatka ommatidiów, kąt międzyommatidialny ~5 stopni
- [x] próbkowanie panoramy equirectangular
- [x] adaptacja czasowa (statyczna scena wygasa)
- [x] `ColumnMap`, który krzyczy zamiast zmyślać retinotopię
- [x] **prawdziwa retinotopia** z tabel Nern 2024 / Matsliah 2024 (892 kolumny)
- [ ] podpiąć `flyvis` jako model płata wzrokowego zamiast naszego high-passa
- [ ] lewe oko: tabele pokrywają tylko prawy płat, `mirror=True` to założenie

### Etap 3 - odczyt ruchowy (ZROBIONE)

- [x] DNa01 / DNa02 / MDN / DNp09 / GF jako API muchy
- [x] skręt z różnicy lewa-prawa w DNa02
- [x] zatrzask ucieczki na włóknie olbrzymim
- [x] generator chodu trójnożnego, jawnie oddzielony od konektomu
- [ ] kalibracja `reference_rate` na realnych częstotliwościach wyładowań

### Etap 4 - ciało (W TOKU)

- [x] zastępnik kinematyczny + protokół `Body`
- [x] adapter NeuroMechFly v2 pod **API FlyGym 2.x** (stare 1.x jest martwe)
- [ ] **nieuruchomiony** - wymaga Pythona 3.12 i MuJoCo, tu jest 3.11
- [ ] zainstalować MuJoCo i przejść pętlę w realnym ciele
- [ ] porównać: te same DN, dwa ciała - co przeżywa transfer

### Etap 5 - fizyczny hexapod (NIE ZACZĘTE)

- [ ] protokół po WiFi: klatki w górę, 18 kątów serw w dół
- [ ] firmware na Pi: kamera, magistrala serw, odruch bezpieczeństwa
- [ ] mózg 1 kHz, pętla robota 20-50 Hz, bufor i wygładzanie
- [ ] sprzęt: hexapod 18-serwowy (~1,5-2,5 tys. zł), Pi, PCA9685

### Etap 6 - eksperymenty

- [x] test looming z kontrolami bodźcowymi (statyczny, oddalający się)
- [x] baseline nie-konektomowy
- [x] **grafy kontrolne**: przetasowane okablowanie, przenumerowane neurony,
      przetasowane znaki - `flyloop controls`
- [ ] ablacje: wyciąć DNp04/DNp02, zmierzyć czy ucieczka znika
- [ ] zmiana pola widzenia, opóźnienie propriocepcji
- [ ] krzywa czułości na `w_syn`

## Czego używamy cudzego

| warstwa | pakiet | licencja |
|---|---|---|
| retinotopia (892 kolumny) | connectome-interpreter | MIT |
| ciało + siatka ommatidiów | flygym 2.x | - |
| płat wzrokowy | flyvis | MIT |
| metodologia | ommatid | MIT |

Okno wersji: flygym wymaga Pythona >= 3.12, flyvis < 3.13. Pełna instalacja
musi iść na **3.12**.

## Co już wiemy, a nie wiedzieliśmy na starcie

1. **~162 synapsy** trzeba, żeby jeden spike odpalił neuron w spoczynku.
   Pojedyncza krawędź konektomu nie robi nic.
2. **Sumacja czasowa zmienia próg.** Połączenie o połowie tej siły i tak odpala
   cel przy 100 Hz. Jest na to test.
3. **Baseline wygrywa na czas reakcji.** 10 linijek arytmetyki ucieka po 0,13 s,
   model konektomowy po 0,68 s. Ale baseline prawdopodobnie fałszywie alarmuje
   na sam widok obiektu. To jest pierwsze realne pytanie badawcze projektu:
   czy różnica to lepsza selektywność, czy tylko wolniejszy obwód?
4. **Binarny wskaźnik ucieczki niczego nie różnicuje.** Trzy z czterech grafów
   uciekają w 100% prób. Dopiero szczytowa częstotliwość wyładowań rozdziela je
   3,16-krotnie. Kontrole znalazły błąd w naszym własnym eksperymencie.
5. **DNp09 i włókno olbrzymie mogą milczeć.** Na realnych danych ommatid zmierzył
   je na 0 Hz; sygnał nieśli DNp04 i DNp02. Nasz odczyt raportuje teraz, która
   populacja odpaliła, zamiast tego zakładać.

## Trzy pułapki

1. **Znaki wag.** Glutaminian u muszki jest hamujący. Najczęstszy cichy błąd.
2. **Skale czasowe.** Mózg 0,1 ms, ciało 10 ms, serwa 50 Hz. Pomieszanie tego
   daje oscylacje bez widocznej przyczyny.
3. **Dostrajanie aż zadziała.** Jeśli dostroisz odczyt na tyle, żeby robot
   chodził, to chodzi odczyt, nie mucha. Dlatego baseline zostaje w repo.
