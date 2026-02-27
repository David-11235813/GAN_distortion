# GAN_distortion
Projekt inżynierski:
"Generatywna sieć neuronowa typu GAN do zniekształcania obrazów na podstawie przekształceń afinicznych w przestrzeni cech"
("A Generative Adversarial Network for Synthesizing Affine-Transformed Images in Feature Space")


### Cel pracy
Celem pracy jest zaprojektowanie, zaimplementowanie i przetestowanie dwuetapowego modelu generatywnej sieci przeciwstawnej (GAN), który generuje obrazy o sensownym wyglądzie wizualnym poprzez wykonywanie przekształceń afinicznych w przestrzeni cech reprezentującej obrazy (definicja "sensowności" jest luźna i robocza, bazująca na znalezionych bazach danych oznaczonych w ten sposób)


### Założenia:
- Obraz traktowany jest jako wektor cech, w którym każda współrzędna odpowiada wartości piksela w położeniu (u,v).
- Przekształcenia obrazów polegają na działaniach afinicznych w przestrzeniach o znacznie mniejszej liczbie wymiarów niż oryginalna przestrzeń obrazu.
- W tej przestrzeni definiujemy zestaw wektorów w1,...,wn, a konkretna transformacja obrazu jest kombinacją liniową tych wektorów: ∑_𝑖 𝜆_i⋅𝑤_i


### Architektura systemu:
- Generator pierwszego poziomu:
Dobiera kierunki deformacji (wektory w przestrzeni cech w_i).
- Generator drugiego poziomu:
Dobiera wartości skalujące (𝜆_𝑖) dla wcześniej ustalonych wektorów deformacji.
- Dyskryminator pierwszego poziomu:
Ocena, czy wygenerowany obraz (dla 𝜆𝑖=1) jest wystarczająco podobny do oryginalnych zdjęć. Można tu wykorzystać np. SSIM, PSNR lub embeddingi z modeli takich jak CLIP.
- Dyskryminator drugiego poziomu:
Ocena „sensowności” obrazu z punktu widzenia percepcji ludzkiej – bazuje na ręcznych adnotacjach lub klasyfikatorze uczonym na takich adnotacjach, najlepiej będzie znaleźć istniejącą sieć odróżniającą obrazy "sensowne" od nie zawierających niczego sensownego.


***


Projekt składa się z dwóch części: GAN_dev (konsolowy UI) i GAN_user (GUI); opis obu z nich znajduje się poniżej.


## GAN_dev - środowisko developerskie

Celem części GAN_dev projektu jest zapewnienie następujących funkcjonalności:
1) ustawienia zasady działania i parametrów generatora (przekształcenia afiniczne w przestrzeni cech)
2) trening modelu sieci GAN z użyciem wybranego datasetu
3) zapis wytrenowanego modelu do folderu middleman_folder
4) plik GAN_dev.ipynb do reprezentacji wyników

## GAN_user - końcowy efekt [tymczasowo nieużywany; do używania i prezentacji służy plik GAN_dev.ipynb z części developerskiej]

Celem części GAN_user projektu jest zapewnienie następujących funkcjonalności:
1) wybór i podgląd obrazu który ma zostać zniekształcony
2) wybór parametrów zniekształcenia (+ modelu wykorzystywanego do wykonania tego zniekształcenia) i wygenerowanie zniekształconego obrazu
3) wizualne i parametryczne przedstawienie wykonanego zniekształcenia (z możliwością porównania efektu końcowego do oryginału)
4) zapis pliku zniekształcenia jako plik .bundle do katalogu zniekształceń
5) odczyt plików .bundle (analogicznie do pkt3)