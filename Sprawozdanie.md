# Wpływ liczby dostępnych etykiet na skuteczność architektur GNN w semi-supervised klasyfikacji węzłów w grafach cytowań

**Autor:** Dominika Bujnarowska

---

## 1. Wprowadzenie

Klasyfikacja węzłów w grafach cytowań jest jednym z kluczowych zadań w uczeniu maszynowym na danych grafowych. W rzeczywistych scenariuszach dostęp do oznaczonych danych jest ograniczony - ręczne przypisywanie etykiet artykułom naukowym jest kosztowne i czasochłonne. Grafowe sieci neuronowe (GNN) są szczególnie dobrze przystosowane do takich warunków, ponieważ propagują informacje przez strukturę grafu, umożliwiając klasyfikację węzłów nawet przy bardzo małej liczbie etykiet.

Artykuły opisujące architektury GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018) oraz GraphSAGE (Hamilton et al., 2017) testowały swoje modele na stałym, standardowym podziale datasetu Cora - 140 oznaczonych węzłów. Nie zbadały systematycznie wpływu liczby etykiet na skuteczność klasyfikacji ani nie porównały wszystkich architektur w kontrolowanych warunkach z różnymi progami etykiet.

Niniejszy projekt wypełnia tę lukę, prowadząc systematyczne porównanie czterech architektur GNN (GCN, GAT, GraphSAGE, APPNP) oraz baselinowego MLP na dwóch datasetach (Cora i Citeseer) przy różnych poziomach dostępności etykiet. Dodatkowe eksperymenty badają odporność modeli na szum w strukturze grafu oraz mechanizm attention w GAT.

---

## 2. Dane

### 2.1 Datasety

| Dataset  | Węzły | Krawędzie | Cechy | Klasy |
|----------|-------|-----------|-------|-------|
| Cora     | 2 708 | 5 429     | 1 433 | 7     |
| Citeseer | 3 327 | 9 104     | 3 703 | 6     |

W obu datasetach węzły reprezentują artykuły naukowe, a krawędzie oznaczają relacje cytowań między nimi. Zadanie polega na klasyfikacji artykułów do kategorii tematycznych na podstawie cech (bag-of-words reprezentacja słów kluczowych) oraz struktury grafu cytowań.

Citeseer jest trudniejszym datasetem ze względu na niższy stosunek krawędzi do węzłów (~2.7 vs ~2.0), co oznacza rzadszy graf i słabszą propagację informacji.

### 2.2 Własne splity treningowe

Oryginalne papiery używają stałego benchmarkowego podziału: 140 węzłów treningowych (~5% danych). W niniejszym projekcie zastosowano własne splity o zadanym procencie etykiet:

- **1%** - ~27 węzłów treningowych (ekstremalny scenariusz semi-supervised)
- **2%** - ~54 węzłów treningowych
- **5%** - ~135 węzłów treningowych (zbliżone do oryginalnego benchmarku)
- **10%** - ~270 węzłów treningowych
- **20%** - ~541 węzłów treningowych

Każdy scenariusz powtórzono 5 razy z różnymi seedami losowości (42, 123, 456, 789, 1000), 
a wyniki raportowane są jako średnia ± odchylenie standardowe.

---

## 3. Modele

### 3.1 GCN - Graph Convolutional Network

Kipf & Welling (2017) zaproponowali uproszczoną operację splotową na grafie, w której każdy węzeł agreguje cechy sąsiadów przez uśrednianie z normalizacją stopniową. Architektura składa się z dwóch warstw GCNConv z funkcją aktywacji ReLU i dropoutem (p=0.5). Benchmark w oryginalnym paperze: ~81.5% accuracy na Corze.

### 3.2 GAT - Graph Attention Network

Veličković et al. (2018) wprowadzili mechanizm attention, który pozwala modelowi uczyć się wag dla każdego sąsiada. Nie wszyscy sąsiedzi są traktowani jednakowo - model przypisuje wyższe wagi połączeniom bardziej informatywnym. Architektura używa 8 attention heads w pierwszej warstwie i 1 w wyjściowej. Benchmark w oryginalnym paperze: ~83.0% accuracy na Corze.

### 3.3 GraphSAGE - Graph Sample and Aggregate

Hamilton et al. (2017) zaproponowali indukcyjne uczenie przez losowe próbkowanie sąsiadów zamiast agregacji wszystkich połączeń. Pozwala to na skalowanie do dużych grafów i klasyfikację węzłów niewidzianych podczas treningu. Benchmark w oryginalnym paperze: ~81.2% accuracy na Corze.

### 3.4 APPNP - Approximate Personalized Propagation of Neural Predictions

Gasteiger et al. (2019) zaproponowali model łączący propagację grafową z ideą PageRank. APPNP rozdziela transformację cech (MLP) od propagacji grafowej. Parametr alfa (α=0.1) kontroluje teleportację - węzeł zachowuje 10% własnych cech w każdym kroku propagacji, co chroni przed nadmiernym wpływem sąsiadów przy małej liczbie etykiet. Benchmark w oryginalnym paperze: ~83.3% accuracy na Corze.

### 3.5 MLP - Multilayer Perceptron (baseline)

Prosty MLP klasyfikujący węzły wyłącznie na podstawie ich własnych cech, bez uwzględnienia struktury grafu. Stanowi dolną granicę porównania - różnica accuracy między GNN a MLP pokazuje wartość informacji zawartej w strukturze cytowań.

---

## 4. Eksperymenty i wyniki

### 4.1 Accuracy vs. liczba etykiet - Cora

| Model     | 1%              | 2%              | 5%              | 10%             | 20%             |
|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|
| GCN       | 0.5534 ± 0.0566 | 0.7188 ± 0.0237 | 0.7922 ± 0.0147 | 0.8152 ± 0.0143 | 0.8274 ± 0.0147 |
| GAT       | 0.5968 ± 0.0580 | 0.7400 ± 0.0179 | 0.8058 ± 0.0090 | 0.8208 ± 0.0087 | 0.8448 ± 0.0102 |
| GraphSAGE | 0.5340 ± 0.0693 | 0.6686 ± 0.0386 | 0.7952 ± 0.0118 | 0.8218 ± 0.0105 | 0.8424 ± 0.0156 |
| APPNP     | 0.6526 ± 0.0647 | 0.7728 ± 0.0339 | 0.8216 ± 0.0101 | 0.8392 ± 0.0126 | 0.8550 ± 0.0113 |
| MLP       | -               | -               | 0.5722 ± 0.0221 | 0.6266 ± 0.0134 | 0.6958 ± 0.0135 |


APPNP osiąga najwyższe wyniki na wszystkich poziomach etykiet, z przewagą +0.0294 nad GCN przy 5% etykiet. GAT jest drugi pod względem accuracy przy 5%, ale charakteryzuje się najniższym odchyleniem standardowym (0.0090), co świadczy o jego stabilności. GCN jest konsekwentnie najsłabszy spośród modeli GNN.

Wartość struktury grafu jest bardzo wyraźna - APPNP przewyższa MLP o +0.2494 przy 5% etykiet. Przewaga maleje wraz z liczbą etykiet (+0.1592 przy 20%), co sugeruje że struktura grafu jest najbardziej wartościowa właśnie w warunkach niedoboru etykiet.

Po analizie wyników przy 5–20% etykiet przeprowadzono dodatkowe eksperymenty 
przy 1% i 2% etykiet w celu zbadania zachowania modeli w warunkach ekstremalnego 
niedoboru danych. Przy 1% etykiet (~27 węzłów treningowych) wszystkie modele 
wykazują dramatyczny spadek accuracy - od -0.169 (APPNP) do -0.261 (GraphSAGE) 
względem 5%. Jednocześnie odchylenie standardowe rośnie do ~0.06, co oznacza 
że przy tak małej liczbie węzłów treningowych dobór konkretnych przykładów 
(seed) ma dominujący wpływ na wynik.

Przy 2% (~54 węzłach) modele zachowują jeszcze 87–94% swojej skuteczności 
z poziomu 5%, co sugeruje że próg krytyczny leży między 1% a 2% etykiet. 
Ranking modeli przy 1% jest spójny z rankingiem przy 5% - APPNP pozostaje 
najbardziej odporny (0.6526), a GraphSAGE degraduje się najbardziej (0.5340).

### 4.2 Accuracy vs. liczba etykiet - Citeseer

| Model     | 5%              | 10%             | 20%             |
|-----------|-----------------|-----------------|-----------------|
| GCN       | 0.6746 ± 0.0206 | 0.6900 ± 0.0177 | 0.6926 ± 0.0140 |
| GAT       | 0.6900 ± 0.0165 | 0.7106 ± 0.0114 | 0.7292 ± 0.0109 |
| GraphSAGE | 0.7012 ± 0.0067 | 0.7182 ± 0.0140 | 0.7352 ± 0.0055 |
| APPNP     | 0.6902 ± 0.0154 | 0.7056 ± 0.0115 | 0.7202 ± 0.0157 |

Na Citeseer ranking modeli zmienia się istotnie. GraphSAGE, który na Corze zajmował trzecie miejsce, na Citeseer wygrywa konsekwentnie na wszystkich poziomach etykiet. APPNP, najlepszy na Corze, spada na drugie lub trzecie miejsce. Ogólne accuracy jest niższe o 10-13 punktów procentowych w porównaniu z Corą, co wynika z rzadszej struktury grafu Citeseer.

### 4.3 Porównanie Cora vs. Citeseer

| Model     | Cora (5%) | Citeseer (5%) | Różnica |
|-----------|-----------|---------------|---------|
| GCN       | 0.7922    | 0.6746        | -0.1176 |
| GAT       | 0.8058    | 0.6900        | -0.1158 |
| GraphSAGE | 0.7952    | 0.7012        | -0.0940 |
| APPNP     | 0.8216    | 0.6902        | -0.1314 |

GraphSAGE wykazuje najmniejszy spadek accuracy między datasetami (-0.0940 przy 5%), podczas gdy APPNP traci najwięcej (-0.1314). Sugeruje to że losowe próbkowanie sąsiadów w GraphSAGE jest bardziej uniwersalną strategią, niezależną od gęstości grafu.

### 4.4 Per-class accuracy - Cora

Analiza per-class ujawniła istotne różnice między klasami:

**Największy przyrost accuracy (20% vs 5% etykiet, uśredniony po modelach):**
- Genetic_Algorithms: +0.2610 - klasa trudna przy małej liczbie etykiet, ale mocno zyskująca
- Rule_Learning: +0.1092
- Case_Based: +0.0712

**Klasy stabilne (mały przyrost):**
- Neural_Networks: +0.0087 - już przy 5% etykiet klasyfikowana z accuracy ~90%
- Theory: -0.0508 - paradoksalnie pogarsza się przy większej liczbie etykiet

GraphSAGE wykazuje dramatyczny przyrost dla Genetic_Algorithms (+0.4835 z 0.34 do 0.82), co sugeruje że próbkowanie sąsiadów szczególnie dobrze radzi sobie z tą klasą gdy dostępne jest więcej danych.

### 4.5 Analiza macierzy pomyłek (5% etykiet, Cora)

| Model     | Accuracy | Najczęstszy błąd                              |
|-----------|----------|-----------------------------------------------|
| GCN       | 0.8040   | Case_Based → Theory (17 przypadków)           |
| GAT       | 0.8070   | Probabilistic → Case_Based (19 przypadków)    |
| GraphSAGE | 0.7720   | Genetic_Alg → Probabilistic (33 przypadków)   |
| APPNP     | 0.8270   | Probabilistic → Case_Based (26 przypadków)    |

Najczęstsze pomyłki dotyczą par klas tematycznie bliskich: Probabilistic_Methods ↔ Case_Based oraz Genetic_Algorithms → Probabilistic_Methods. GraphSAGE przy 5% etykiet ma szczególnie słabą klasyfikację Genetic_Algorithms (recall 0.34), co tłumaczy jego gorsze wyniki przy małej liczbie etykiet.

### 4.6 Analiza wag attention - GAT

Histogram wag attention w pierwszej warstwie GAT ujawnia bimodalny rozkład - większość krawędzi skupia się wokół wartości ~0.15 (mało informatywne cytowania), a druga grupa osiąga wagi ~0.50 (kluczowe połączenia). Średnia waga wynosi 0.2042.

Spośród 10 krawędzi z najwyższą wagą attention, wszystkie 10 (100%) łączyły węzły tej samej klasy tematycznej. Wynik ten potwierdza że GAT nauczył się skutecznie identyfikować cytowania wewnątrz tej samej dziedziny jako bardziej informatywne dla klasyfikacji.

### 4.7 Odporność na szum w strukturze grafu

| Model     | 0% szumu | 10%    | 30%    | 50%    | 70%    | Spadek 50% |
|-----------|----------|--------|--------|--------|--------|------------|
| GCN       | 0.7943   | 0.7887 | 0.7640 | 0.7387 | 0.6610 | -0.0557    |
| GAT       | 0.7957   | 0.7987 | 0.7620 | 0.7430 | 0.6843 | -0.0527    |
| GraphSAGE | 0.7640   | 0.7570 | 0.7130 | 0.6737 | 0.6183 | -0.0903    |
| APPNP     | 0.8250   | 0.8053 | 0.7787 | 0.7370 | 0.6377 | -0.0880    |

GAT wykazuje najwyższą odporność na usuwanie krawędzi - przy 10% szumu accuracy wręcz nieznacznie rośnie (+0.0030), a przy 50% spada o zaledwie 0.0527. Mechanizm attention pozwala modelowi ignorować mniej wartościowe połączenia, kompensując ich losowe usuwanie.

GraphSAGE i APPNP są bardziej wrażliwe na szum (spadki -0.0903 i -0.0880 przy 50%). W przypadku APPNP, choć teleportacja chroni przed całkowitą utratą własnych cech węzła, silny spadek przy 70% szumu (-0.1873) sugeruje że model jest zależny od spójności struktury grafu.

---

## 5. Wnioski

### 5.1 Ranking modeli

Na datasecie Cora przy 5% etykiet ranking modeli prezentuje się następująco:

**APPNP > GAT > GraphSAGE > GCN**

APPNP dominuje dzięki mechanizmowi teleportacji, który chroni reprezentacje węzłów przed nadmiernym wpływem sąsiadów - szczególnie wartościowe przy małej liczbie etykiet. GAT zajmuje drugie miejsce pod względem accuracy i pierwsze pod względem stabilności.

### 5.2 Ranking zależy od datasetu

Na datasecie Citeseer ranking zmienia się na:

**GraphSAGE > GAT > APPNP > GCN**

GraphSAGE, który na Corze zajmował trzecie miejsce, na rzadszym grafie Citeseer wygrywa konsekwentnie. Losowe próbkowanie sąsiadów okazuje się bardziej universalną strategią niż teleportacja PageRank. Sugeruje to że dobór architektury GNN powinien uwzględniać gęstość grafu.

### 5.3 Wartość struktury grafu

Wszystkie modele GNN znacząco przewyższają MLP - przy 5% etykiet przewaga wynosi od +0.2200 (GCN) do +0.2494 (APPNP). Przewaga maleje przy 20% etykiet (od +0.1316 do +0.1592), co potwierdza że struktura grafu jest najcenniejsza właśnie w warunkach niedoboru etykiet. Gdy etykiet jest więcej, cechy węzłów stają się relatywnie ważniejsze.

### 5.4 Stabilność

GAT wykazuje najniższe odchylenie standardowe przy 5% etykiet (0.0090), będąc prawie dwukrotnie bardziej stabilny niż GCN (0.0147). Mechanizm attention redukuje wrażliwość na losowy dobór węzłów treningowych.

### 5.5 Odporność na szum

GAT jest najbardziej odpornym modelem na szum w strukturze grafu - przy usunięciu 50% krawędzi traci jedynie 0.0527 punktu accuracy. Mechanizm attention naturalnie kompensuje szum przez obniżenie wag uszkodzonych połączeń.

### 5.6 Wkład badawczy

Oryginalne papiery (Kipf & Welling 2017, Veličković 2018, Hamilton 2017, Gasteiger 2019) testowały modele wyłącznie na standardowym benchmarku i nie odpowiadały na pytanie jak liczba etykiet wpływa na skuteczność poszczególnych architektur. Niniejszy projekt wprowadza:

1. Systematyczne porównanie 4 architektur GNN przy 5 poziomach dostępności 
etykiet (1%, 2%, 5%, 10%, 20%), powtórzone 5-krotnie dla wiarygodności 
statystycznej, z identyfikacją progu krytycznego między 1% a 2% etykiet
2. Analizę na dwóch datasetach, ujawniającą że ranking modeli nie jest uniwersalny
3. Eksperymenty z szumem w strukturze grafu jako nowy wymiar porównania
4. Analizę wag attention potwierdzającą że GAT uczy się faworyzować cytowania wewnątrzdyscyplinarne

---

## 6. Literatura

1. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR 2018.
3. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS 2017.
4. Gasteiger, J., Bojchevski, A., & Günnemann, S. (2019). *Predict then Propagate: Graph Neural Networks meet Personalized PageRank*. ICLR 2019.
5. Yang, Z., Cohen, W., & Salakhudinov, R. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings*. ICML 2016. (Cora/Citeseer dataset split)
